#!/bin/bash
#SBATCH --job-name=data-proc
#SBATCH --partition=gpu-2080ti
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

python src/preprocess_utils/preprocess_wmt_train_data.py 2017 mul_ss
python src/preprocess_utils/preprocess_wmt_train_data.py 2018 mul_ss
python src/preprocess_utils/preprocess_wmt_train_data.py 2019 mul_ss
python src/preprocess_utils/preprocess_wmt_train_data.py 2020 mul_ss
# python src/preprocess_utils/preprocess_wmt_train_data.py 2017 one_ss_random_span
# python src/preprocess_utils/preprocess_wmt_train_data.py 2018 one_ss_random_span
# python src/preprocess_utils/preprocess_wmt_train_data.py 2019 one_ss_random_span