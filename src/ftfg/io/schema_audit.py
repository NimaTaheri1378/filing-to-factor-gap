from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from ftfg.config import load_yaml
from ftfg.manifests import write_manifest

LOG = logging.getLogger(__name__)


def _connect_wrds():
    try:
        import wrds
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("Install the wrds extra to run schema audits: pip install -e '.[wrds]'") from exc
    return wrds.Connection()


def audit_wrds_schema(config_path: str | Path, out_dir: str | Path) -> Path:
    cfg = load_yaml(config_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    db = _connect_wrds()
    visible = set(db.list_libraries())
    records: list[dict[str, Any]] = []
    for layer, spec in cfg["wrds_libraries"].items():
        for lib in spec.get("libraries", []):
            if lib not in visible:
                records.append({"layer": layer, "library": lib, "visible": False, "table": None})
                continue
            try:
                tables = db.list_tables(library=lib)
            except Exception as exc:
                records.append({"layer": layer, "library": lib, "visible": True, "error": str(exc), "table": None})
                continue
            for table in tables:
                if _table_matches(table, spec.get("table_patterns", [])):
                    try:
                        desc = db.raw_sql(
                            """
                            select column_name
                            from information_schema.columns
                            where table_schema = %(schema)s
                              and table_name = %(table)s
                            order by ordinal_position
                            """,
                            params={"schema": lib, "table": table},
                        )
                        cols = ",".join(desc["column_name"].astype(str).tolist())
                    except Exception as exc:
                        cols = f"DESCRIBE_ERROR:{exc}"
                    records.append({"layer": layer, "library": lib, "visible": True, "table": table, "columns": cols})
    schema = pd.DataFrame(records)
    schema_path = out / "schema_audit.csv"
    schema.to_csv(schema_path, index=False)
    write_manifest(
        out / "schema_audit_manifest.json",
        {
            "kind": "schema_audit",
            "visible_library_count": len(visible),
            "layers": sorted(cfg["wrds_libraries"].keys()),
            "records": len(records),
        },
    )
    LOG.info("Wrote schema audit to %s", schema_path)
    db.close()
    return schema_path


def _table_matches(table: str, patterns: list[str]) -> bool:
    low = table.lower()
    for pattern in patterns:
        terms = [term for term in pattern.lower().split("%") if term]
        if terms and all(term in low for term in terms):
            return True
    return False
