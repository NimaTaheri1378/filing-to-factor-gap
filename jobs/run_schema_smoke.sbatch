#!/bin/bash
#SBATCH --job-name=ftfg_schema_smoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=artifacts/logs/%x_%j.out
#SBATCH --error=artifacts/logs/%x_%j.err

set -euo pipefail
cd "/scratch/nt612/Github/Filing-to-Factor Gap"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
export FTFG_SEC_USER_AGENT="${FTFG_SEC_USER_AGENT:-filing-to-factor-gap research contact@example.com}"
"$HOME/.conda/envs/ml_core/bin/python" -m ftfg.cli wrds-smoke --out-dir artifacts/private/wrds_smoke
"$HOME/.conda/envs/ml_core/bin/python" -m ftfg.cli schema-audit --schema-config configs/schema_targets.yml
"$HOME/.conda/envs/ml_core/bin/python" -m ftfg.cli pull-wrds --smoke-year 2021 --no-daily
"$HOME/.conda/envs/ml_core/bin/python" -m ftfg.cli smoke --no-deep
