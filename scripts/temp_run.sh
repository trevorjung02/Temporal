#!/bin/bash
#SBATCH --job-name=data-proc
#SBATCH --partition=gpu-rtx6k
#SBATCH --account=cse
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tjung2@uw.edu
#SBATCH --dependency=afterany:6700270

# I use source to initialize conda into the right environment.
cat $0
echo "--------------------"

source ~/.bashrc
conda activate ckl

mv news-docs/news-docs.2007.en.filtered raw_data/wmt
mv news-docs/news-docs.2008.en.filtered raw_data/wmt