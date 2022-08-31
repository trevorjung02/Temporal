# srun --partition=gpu-a40 --account=ark --nodes=1 --cpus-per-task=8 --mem=32G --gres=gpu:1 --time=04:59:00 python run.py --config configs/templama/training/t5_kadapters_2010_prefixed.json

srun --partition=gpu-a40 --account=ark --nodes=1 --cpus-per-task=8 --mem=32G --time=04:59:00 python test.py
