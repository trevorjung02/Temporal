#!/bin/bash
#SBATCH --partition=gpu-a40
#SBATCH --account=cse
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --gres=gpu:2
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

# python src/run.py --config configs/wmt/training/t5_stale_one_ss_random_span.json -lr 1e-5 -datav 2007

# python src/run.py --config configs/streamqa/training/t5_outofbox.json -lr 1e-5

# python src/run.py configs/streamqa/training/t5_outofbox.json -lr 1e-5

python src/run.py --config configs/templama/training/t5_baseline_full.json
# python src/run.py --config configs/wmt/training/t5_stale.json 
