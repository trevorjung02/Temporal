import pandas as pd
import os
import json

# Must have nyt train files split by year. Outputs the combined full dataset.

def main():
    train_output_path = "data/nyt/nyt_train_full.csv"
    val_output_path = "data/nyt/nyt_val_full.csv"
    val_answers_output_path = "data/nyt/nyt_val_full_answers.json"

    if os.path.exists(train_output_path):
        os.remove(train_output_path)
    if os.path.exists(val_output_path):
        os.remove(val_output_path)

    base_train_id = 0
    base_val_id = 0
    ids_to_answers = {}
    for date in range(1981, 2021):
        print(date)
        train = pd.read_csv(f"data/nyt/nyt_train_{date}.csv", encoding='utf-8')
        train['id'] = train['id'] + base_train_id

        val = pd.read_csv(f"data/nyt/nyt_val_{date}.csv", encoding='utf-8')
        val['id'] = val['id'] + base_val_id

        for row in val.itertuples():
            ids_to_answers[str(row.id)] = [row.output]

        base_train_id += len(train)
        base_val_id += len(val)

        train.to_csv(train_output_path, mode='a', header=not os.path.exists(train_output_path), index=False)
        val.to_csv(val_output_path, mode='a', header=not os.path.exists(val_output_path), index=False)
    with open(val_answers_output_path, "w", encoding='utf-8') as f:
        json.dump(ids_to_answers, f, ensure_ascii=False)




if __name__ == "__main__":
    main()