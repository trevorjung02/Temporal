#!/bin/bash
#SBATCH --job-name=data-proc
#SBATCH --partition=gpu-a40
#SBATCH --account=ark
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=1-0:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tjung2@uw.edu
#SBATCH --dependency=afterany:6700270

# I use source to initialize conda into the right environment.
cat $0
echo "--------------------"

source ~/.bashrc
conda activate ckl

python run.py --config configs/templama/training/t5_kadapters_ensemble_mixed.json