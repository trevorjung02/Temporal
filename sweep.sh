#!/bin/bash
#SBATCH --job-name=data-proc
#SBATCH --partition=gpu-a40
#SBATCH --account=ark
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=16:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tjung2@uw.edu

# I use source to initialize conda into the right environment.
cat $0
echo "--------------------"
    
source ~/.bashrc
conda activate ckl

wandb agent --count 1 tjung2/temporal_questions/mli64tqp
wandb agent --count 1 tjung2/temporal_questions/mli64tqp
wandb agent --count 1 tjung2/temporal_questions/mli64tqp