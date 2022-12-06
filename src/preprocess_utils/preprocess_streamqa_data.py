from argparse import ArgumentParser
import csv
import json
import pandas as pd
from datetime import datetime
import os 

def main():
    args = get_args()
    if not args:
        return 

    df = pd.read_json(f"raw_data/streamingqa/streaminqa_{args.split}.jsonl", lines=True)
    datasets = {}
    ids_to_answers = {}
    if args.split == 'test':
        years = [2020]
    else:
        years = list(range(2007, 2020))
    for year in years:
        datasets[str(year)] = []
        ids_to_answers[str(year)] = {}
    max_answer_len = 0
    max_question_len = 0

    for i in range(len(df)):
        row = df.iloc[i]
        date = str(datetime.fromtimestamp(row['question_ts']).year)
        index = len(datasets[date])
        query = row['question']
        if args.split == 'train':
            answer = f"<extra_id_0> {row['answers'][0]} <extra_id_1>"
        else:
            answer = ';'.join(row['answers'])
            ids_to_answers[date][str(index)] = row['answers']
        res = [index, date, query, answer]
        datasets[date].append(res)

        max_question_len = max(max_question_len, len(query.split()))
        max_answer_len = max(max_answer_len, len(answer.split()))

    print(f"max answer length = {max_answer_len}")
    print(f"max query length = {max_question_len}")

    write_datasets(datasets, ids_to_answers, args.split)

def write_datasets(datasets, ids_to_answers, split):
    for year in datasets:
        dataset_len = len(datasets[year])
        print(dataset_len)
        datasets[year] = datasets[year][:dataset_len-dataset_len%32]
        file_name = f"data/streamqa/{split}/{year}.csv"
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        with open(file_name, "w", encoding='utf-8') as csvfile:
            w = csv.writer(csvfile)
            w.writerow(["id", "date", "input", "output"])
            for row in datasets[year]:
                w.writerow(row)
        
        if split != 'train':
            file_name = f"data/streamqa/{split}/{year}_answers.json"
            with open(file_name, "w", encoding='utf-8') as f:
                json.dump(ids_to_answers[year], f, ensure_ascii=False)

    dataset_full = []
    ids_to_answers_full = {}
    for year in datasets: 
        for row in datasets[year]:
            old_index = str(row[0])
            new_index = len(dataset_full)
            new_row = row.copy()
            new_row[0] = new_index
            dataset_full.append(new_row)
            if split != 'train':
                ids_to_answers_full[new_index] = ids_to_answers[year][old_index]

    if split != 'train':
        file_name = f"data/streamqa/{split}/full_answers.json"
        with open(file_name, "w", encoding='utf-8') as f:
            json.dump(ids_to_answers_full, f, ensure_ascii=False)

    file_name = f"data/streamqa/{split}/full.csv"
    with open(file_name, "w", encoding='utf-8') as csvfile:
        w = csv.writer(csvfile)
        w.writerow(["id", "date", "input", "output"])
        for row in dataset_full: 
            w.writerow(row)

def get_args():
    parser = ArgumentParser()
    parser.add_argument('split', default=None, type=str)
    args = parser.parse_args()
    if(args.split not in {'train', 'val', 'test'}):
        print(f"Invalid split: '{args.split}'. Must be one of train, val, test")
        return None
    print(f"Preprocessing streamqa split: {args.split}")
    return args

if __name__ == "__main__":
    main()