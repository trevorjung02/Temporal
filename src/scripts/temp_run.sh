#!/bin/bash
#SBATCH --partition=gpu-a40
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
conda activate temporal

# srun python src/run.py --config configs/wmt/training/t5_stale.json -checkpoint_path outputs/wmtbaseline_full/epoch=0-f1_score=0.1941-em_score=0.0757-v1.ckpt

# python src/preprocess_utils/preprocess_wmt_train_data_full.py

python src/preprocess_utils/preprocess_wmt_train_data_full_gpt.py 2017 -mask_mode causal
