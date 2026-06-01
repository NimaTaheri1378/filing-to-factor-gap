#!/usr/bin/env bash
set -euo pipefail

ROOT="${FTFG_ROOT:-/scratch/nt612/Github/Filing-to-Factor Gap}"
PYTHON="${FTFG_PYTHON:-$HOME/.conda/envs/ml_core/bin/python}"

cd "$ROOT"
mkdir -p artifacts/logs artifacts/manifests data/processed
LOGFILE="${FTFG_RESUME_LOG:-artifacts/logs/resume_after_sec_$(date +%Y%m%d_%H%M%S).log}"
exec >> "$LOGFILE" 2>&1

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"

echo "[$(date -Is)] resume_after_sec: root=$ROOT"
echo "[$(date -Is)] resume_after_sec: python=$PYTHON"
"$PYTHON" -m ftfg.cli build-real-panel
echo "[$(date -Is)] resume_after_sec: real panel complete"
"$PYTHON" -m ftfg.cli full --include-deep
echo "[$(date -Is)] resume_after_sec: full outputs complete"
