import pandas as pd
import os
import json
from argparse import ArgumentParser

# Must have wmt train files split by year. Outputs the combined full dataset.

def main():
    parser = ArgumentParser()
    parser.add_argument('end_year', type=int)
    parser.add_argument('-mask_mode', type=str)
    args = parser.parse_args()

    if args.mask_mode is None:
        args.mask_mode = ''
    else:
        args.mask_mode = "_" + args.mask_mode

    train_output_path = f"data/wmt{args.mask_mode}/wmt_train_full.csv"
    val_output_path = f"data/wmt{args.mask_mode}/wmt_val_full.csv"

    if os.path.exists(train_output_path):
        os.remove(train_output_path)
    if os.path.exists(val_output_path):
        os.remove(val_output_path)

    base_train_id = 0
    base_val_id = 0
    yearly_val_size = 500
    for date in range(2007, 2020):
        print(date)
        train = pd.read_csv(f"data/wmt{args.mask_mode}/wmt_train_{date}.csv", encoding='utf-8')
        train['id'] = train['id'] + base_train_id

        val = pd.read_csv(f"data/wmt{args.mask_mode}/wmt_val_{date}.csv", encoding='utf-8', nrows=yearly_val_size)
        val['id'] = val['id'] + base_val_id

        base_train_id += len(train)
        base_val_id += len(val)

        train.to_csv(train_output_path, mode='a', header=not os.path.exists(train_output_path), index=False)
        val.to_csv(val_output_path, mode='a', header=not os.path.exists(val_output_path), index=False)

if __name__ == "__main__":
    main()