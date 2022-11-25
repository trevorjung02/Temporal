# srun --partition=gpu-a40 --account=ark --nodes=1 --cpus-per-task=8 --mem=32G --gres=gpu:1 --time=04:59:00 python run.py --config configs/templama/training/t5_kadapters_2010_prefixed.json

# srun --partition=gpu-rtx6k --account=cse --nodes=1 --cpus-per-task=8 --mem=32G --time=1-0:00:00 wget https://data.statmt.org/news-crawl/doc/en/news-docs.2020.en.filtered.gz --user newscrawl --password acrawl4me

# srun --partition=gpu-rtx6k --account=cse --nodes=1 --cpus-per-task=8 --gres=gpu:1 --mem=32G --time=01:59:00 python test.py

# srun --partition=gpu-rtx6k --account=cse --nodes=1 --cpus-per-task=8 --mem=32G --time=04:59:00 python peek.py

# srun --partition=gpu-rtx6k --account=cse --nodes=1 --cpus-per-task=1 --gres=gpu:1 --mem=48G --time=1-0:00:00 python extract_adapter.py outputs/templamakadapter_2010_2freeze_158_128/epoch=9-f1_score=0.123-em_score=0.029.ckpt

# srun --partition=gpu-rtx6k --account=cse --nodes=1 --cpus-per-task=1 --gres=gpu:1 --mem=48G --time=1-0:00:00 python load_adapter.py --config configs/templama/training/t5_kadapters_yearly_large.json -checkpoint_path outputs/templamakadapter_2010_2freeze_158_128/epoch=9-f1_score=0.123-em_score=0.029.ckpt

# srun --partition=gpu-rtx6k --account=cse --nodes=1 --cpus-per-task=1 --gres=gpu:1 --mem=48G --time=1-0:00:00 python extract_adapter.py outputs/wmtkadapter_2017_2freeze_11221222324_128/epoch=0-f1_score=0.1907-em_score=0.1624.ckpt

# srun --partition=gpu-a40 --account=cse --nodes=1 --cpus-per-task=1 --gres=gpu:1 --mem=48G --time=1-0:00:00 python run.py --config configs/wmt/training/t5_kadapters_ensemble.json -datav 2017 

srun --partition=gpu-rtx6k --account=cse --nodes=1 --cpus-per-task=8 --gres=gpu:1 --mem=48G --time=1-0:00:00 python run.py --config configs/templama/training/t5_kadapters_ensemble.json