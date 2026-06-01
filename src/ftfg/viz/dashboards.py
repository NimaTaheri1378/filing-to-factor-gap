from __future__ import annotations

from pathlib import Path


def write_figure_gallery(figures: dict[str, Path], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Figure Gallery", ""]
    for name, path in sorted(figures.items()):
        rel = path.as_posix()
        lines.append(f"## {name.replace('_', ' ').title()}")
        lines.append("")
        lines.append(f"![{name}]({rel})")
        lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")
