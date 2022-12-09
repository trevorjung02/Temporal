#!/bin/bash
#SBATCH --job-name=data-proc
#SBATCH --partition=ckpt
#SBATCH --account=cse
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=1-0:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tjung2@uw.edu
#SBATCH --dependency=afterany:6700270
#SBATCH --output=slurm_new/slurm-%j.out

# I use source to initialize conda into the right environment.
cat $0
echo "--------------------"

source ~/.bashrc
conda activate ckl

python src/run.py --config configs/templama/training/t5_stale.json -lr 1e-5 -checkpoint_path outputs/wmtbaseline_full/epoch=0-f1_score=0.2586-em_score=0.2194.ckpt
