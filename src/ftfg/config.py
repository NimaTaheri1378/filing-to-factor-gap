from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    data_root: Path
    artifact_root: Path
    manifest_dir: Path
    public_table_dir: Path
    public_figure_dir: Path

    def ensure(self) -> None:
        for path in [
            self.data_root,
            self.artifact_root,
            self.manifest_dir,
            self.public_table_dir,
            self.public_figure_dir,
        ]:
            path.mkdir(parents=True, exist_ok=True)


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return data


def load_config(path: str | Path = "configs/sample_period.yml") -> dict[str, Any]:
    return load_yaml(path)


def project_paths(config: dict[str, Any], root: str | Path | None = None) -> ProjectPaths:
    root_path = Path(root or os.getcwd()).resolve()
    outputs = config.get("outputs", {})
    data_root = Path(os.getenv("FTFG_DATA_ROOT", outputs.get("data_root", "data")))
    artifact_root = Path(os.getenv("FTFG_ARTIFACT_ROOT", outputs.get("artifact_root", "artifacts")))
    manifest_dir = Path(outputs.get("manifest_dir", artifact_root / "manifests"))
    table_dir = Path(outputs.get("public_table_dir", artifact_root / "tables" / "public"))
    figure_dir = Path(outputs.get("public_figure_dir", artifact_root / "figures" / "public"))

    def absolutize(path: Path) -> Path:
        return path if path.is_absolute() else root_path / path

    return ProjectPaths(
        root=root_path,
        data_root=absolutize(data_root),
        artifact_root=absolutize(artifact_root),
        manifest_dir=absolutize(manifest_dir),
        public_table_dir=absolutize(table_dir),
        public_figure_dir=absolutize(figure_dir),
    )
