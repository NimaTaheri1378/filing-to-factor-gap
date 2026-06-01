#!/bin/bash
#SBATCH --job-name=ftfg_data_smoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=artifacts/logs/%x_%j.out
#SBATCH --error=artifacts/logs/%x_%j.err

set -euo pipefail
cd "/scratch/nt612/Github/Filing-to-Factor Gap"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
export FTFG_SEC_USER_AGENT="${FTFG_SEC_USER_AGENT:-Filing-to-Factor-Gap research contact@example.com}"
"$HOME/.conda/envs/ml_core/bin/python" -m ftfg.cli pull-wrds --smoke-year 2021 --no-daily
"$HOME/.conda/envs/ml_core/bin/python" -m ftfg.cli pull-sec --max-ciks 25 --force
"$HOME/.conda/envs/ml_core/bin/python" -m ftfg.cli build-real-panel
"$HOME/.conda/envs/ml_core/bin/python" -m ftfg.cli full --include-deep
