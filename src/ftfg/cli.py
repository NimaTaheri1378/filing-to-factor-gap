from __future__ import annotations

import argparse
import logging
import shutil

from ftfg.config import load_config, project_paths
from ftfg.features.real_panel import build_real_analysis_panel
from ftfg.io.schema_audit import audit_wrds_schema
from ftfg.io.sec_edgar import fetch_sec_companyfacts_for_companies
from ftfg.io.wrds_extract import extract_wrds_core, wrds_smoke
from ftfg.logging_utils import configure_logging
from ftfg.pipeline import build_analysis_panel, run_outputs

LOG = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ftfg")
    parser.add_argument("--config", default="configs/sample_period.yml")
    sub = parser.add_subparsers(dest="command", required=True)

    smoke = sub.add_parser("smoke")
    smoke.add_argument("--synthetic", action="store_true", default=True)
    smoke.add_argument("--no-deep", action="store_true")

    sub.add_parser("build-features")
    sub.add_parser("make-tables")
    sub.add_parser("make-figures")
    full = sub.add_parser("full")
    full.add_argument("--synthetic", action="store_true", default=False)
    full.add_argument("--include-deep", action="store_true", default=True)

    schema = sub.add_parser("schema-audit")
    schema.add_argument("--schema-config", default="configs/schema_targets.yml")

    wrds = sub.add_parser("wrds-smoke")
    wrds.add_argument("--out-dir", default="artifacts/private/wrds_smoke")

    pull = sub.add_parser("pull-wrds")
    pull.add_argument("--smoke-year", type=int)
    pull.add_argument("--no-daily", action="store_true")
    pull.add_argument("--force", action="store_true")

    build_real = sub.add_parser("build-real-panel")
    build_real.add_argument("--synthetic", action="store_true", default=False)

    sec = sub.add_parser("pull-sec")
    sec.add_argument("--max-ciks", type=int)
    sec.add_argument("--force", action="store_true")

    sub.add_parser("clean")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    paths = project_paths(cfg)
    paths.ensure()
    configure_logging(paths.artifact_root / "logs" / f"{args.command}.log")

    if args.command == "smoke":
        run_outputs(args.config, synthetic=True, include_deep=not args.no_deep)
    elif args.command == "build-features":
        build_analysis_panel(args.config, synthetic=True)
    elif args.command in {"make-tables", "make-figures"}:
        run_outputs(args.config, synthetic=True, include_deep=False)
    elif args.command == "full":
        run_outputs(args.config, synthetic=args.synthetic, include_deep=args.include_deep)
    elif args.command == "schema-audit":
        audit_wrds_schema(args.schema_config, paths.manifest_dir / "schema_audit")
    elif args.command == "wrds-smoke":
        wrds_smoke(args.out_dir)
    elif args.command == "pull-wrds":
        sample = cfg["sample"]
        if args.smoke_year:
            years = [args.smoke_year]
            start = f"{args.smoke_year}-01-01"
            end = f"{args.smoke_year}-12-31"
            smoke = True
        else:
            start = sample["warmup_start"]
            end = sample["holdout_end"]
            years = list(range(int(start[:4]), int(end[:4]) + 1))
            smoke = False
        extract_wrds_core(
            paths.data_root,
            start=start,
            end=end,
            years=years,
            smoke=smoke,
            include_daily=not args.no_daily,
            force=args.force,
        )
    elif args.command == "pull-sec":
        sample = cfg["sample"]
        fetch_sec_companyfacts_for_companies(
            paths.data_root / "raw" / "wrds" / "compustat" / "company.parquet",
            paths.data_root / "raw" / "sec",
            start=sample["warmup_start"],
            end=sample["holdout_end"],
            max_ciks=args.max_ciks,
            force=args.force,
        )
    elif args.command == "build-real-panel":
        if args.synthetic:
            build_analysis_panel(args.config, synthetic=True)
        else:
            build_real_analysis_panel(paths.data_root, paths.manifest_dir)
    elif args.command == "clean":
        for rel in ["data/synthetic", "data/processed", "artifacts/tables/public", "artifacts/figures/public"]:
            target = paths.root / rel
            if target.exists():
                shutil.rmtree(target)
                LOG.info("Removed %s", target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
