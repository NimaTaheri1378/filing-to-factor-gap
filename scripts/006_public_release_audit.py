from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

FORBIDDEN_PATHS = re.compile(
    r"(^data/|^site/|^\.codex/|^\.env$|artifacts/private|artifacts/logs|artifacts/remote|\.pgpass|\.parquet$|\.zip$)"
)
SECRET_PATTERNS = re.compile(
    r"(BEGIN .*PRIVATE KEY|postgresql://|PGPASSWORD|WRDS_PASSWORD|password\s*=|secret\s*=|token\s*=|api[_-]?key\s*=)",
    re.IGNORECASE,
)


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True, encoding="utf-8", errors="replace")


def main() -> int:
    staged = [line.strip() for line in _git("ls-files", "--cached").splitlines()]
    bad_paths = [path for path in staged if FORBIDDEN_PATHS.search(path.replace("\\", "/"))]
    if bad_paths:
        print("Forbidden staged paths:")
        print("\n".join(bad_paths))
        return 1

    allow_pattern_files = {".env.example", "scripts/006_public_release_audit.py"}
    searchable = [
        path
        for path in staged
        if Path(path).suffix.lower() not in {".png", ".pdf", ".svg"}
        and path.replace("\\", "/") not in allow_pattern_files
    ]
    bad_hits: list[str] = []
    for path in searchable:
        text = Path(path).read_text(encoding="utf-8", errors="ignore") if Path(path).exists() else ""
        if SECRET_PATTERNS.search(text):
            bad_hits.append(path)
    if bad_hits:
        print("Potential secret patterns found:")
        print("\n".join(bad_hits))
        return 1

    print("Public release audit passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
