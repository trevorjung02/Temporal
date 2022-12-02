import pandas as pd
import os
import json

# Must have wmt train files split by year. Outputs the combined full dataset.

def main():
    train_output_path = "data/wmt/wmt_train_full.csv"
    val_output_path = "data/wmt/wmt_val_full.csv"
    val_answers_output_path = "data/wmt/wmt_val_full_answers.json"

    if os.path.exists(train_output_path):
        os.remove(train_output_path)
    if os.path.exists(val_output_path):
        os.remove(val_output_path)

    base_train_id = 0
    base_val_id = 0
    ids_to_answers = {}
    yearly_val_size = 1000
    for date in range(2007, 2020):
        print(date)
        train = pd.read_csv(f"data/wmt/wmt_train_{date}.csv", encoding='utf-8')
        train['id'] = train['id'] + base_train_id

        val = pd.read_csv(f"data/wmt/wmt_val_{date}.csv", encoding='utf-8', nrows=yearly_val_size)
        val['id'] = val['id'] + base_val_id

        mask_len = len("<extra_id_0> ")
        for row in val.itertuples():
            ids_to_answers[str(row.id)] = [row.output[mask_len:-mask_len]]
            # print(row.output[mask_len:-mask_len])

        base_train_id += len(train)
        base_val_id += len(val)

        train.to_csv(train_output_path, mode='a', header=not os.path.exists(train_output_path), index=False)
        val.to_csv(val_output_path, mode='a', header=not os.path.exists(val_output_path), index=False)
    with open(val_answers_output_path, "w", encoding='utf-8') as f:
        json.dump(ids_to_answers, f, ensure_ascii=False)

if __name__ == "__main__":
    main()