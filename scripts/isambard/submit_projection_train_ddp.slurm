#!/bin/bash
#SBATCH -A brics.u6dr
#SBATCH -p workq
#SBATCH -N 2
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH -J graphmed_train
#SBATCH -o logs/graphmed_train_%j.out

set -euo pipefail

cd "${GRAPHMED_HOME:-$HOME/GraphMed-LT}"
mkdir -p logs save_model
bash scripts/isambard/run_projection_train_ddp.sh
