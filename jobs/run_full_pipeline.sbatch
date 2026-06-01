#!/bin/bash
#SBATCH --job-name=ftfg_full
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=3-00:00:00
#SBATCH --output=artifacts/logs/%x_%j.out
#SBATCH --error=artifacts/logs/%x_%j.err

set -euo pipefail
cd "/scratch/nt612/Github/Filing-to-Factor Gap"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
export FTFG_SEC_USER_AGENT="${FTFG_SEC_USER_AGENT:-filing-to-factor-gap research contact@example.com}"
"$HOME/.conda/envs/ml_core/bin/python" -m ftfg.cli pull-wrds --force
"$HOME/.conda/envs/ml_core/bin/python" -m ftfg.cli pull-sec
"$HOME/.conda/envs/ml_core/bin/python" -m ftfg.cli build-real-panel
"$HOME/.conda/envs/ml_core/bin/python" -m ftfg.cli full --include-deep
