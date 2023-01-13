#!/bin/bash
#SBATCH --partition=gpu-a40
#SBATCH --account=cse
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:4
#SBATCH --time=1-0:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tjung2@uw.edu
#SBATCH --dependency=afterany:6700270
#SBATCH --output=slurm_new/slurm-%j.out

# I use source to initialize conda into the right environment.
cat $0
echo "--------------------"

source ~/.bashrc
conda activate temporal

# PL_FAULT_TOLERANT_TRAINING=1 srun python src/run.py --config configs/wmt/training/t5_stale.json -checkpoint_path outputs/wmtbaseline_full/epoch=0-f1_score=0.2211-em_score=0.0997-v1.ckpt
# PL_FAULT_TOLERANT_TRAINING=1 srun python src/run.py --config configs/wmt/training/t5_stale.json 
# PL_FAULT_TOLERANT_TRAINING=1 srun python src/run.py --config configs/wmt/training/t5_stale_16.json
# srun python src/run.py --config configs/templama/training/t5_stale.json -checkpoint_path outputs/wmtbaseline_full/epoch=0-f1_score=0.2211-em_score=0.0997-v1.ckpt
# srun python src/run.py --config configs/streamqa/training/t5_stale.json -checkpoint_path outputs/wmtbaseline_full/epoch=0-f1_score=0.2211-em_score=0.0997-v1.ckpt

PL_FAULT_TOLERANT_TRAINING=1 srun python src/run.py --config configs/wmt/training/gpt/gpt2_stale.json