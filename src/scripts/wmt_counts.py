import argparse
import os
import pandas as pd
import json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("year", type=str)
    args = parser.parse_args()

    if args.year.isdigit():
        count_year(args.year)
    elif args.year == 'all':
        for year in range(2007, 2021):
            count_year(year)

def count_year(year):
    path = f"data/wmt/wmt_train_{year}.csv"
    df = pd.read_csv(path, usecols=[2,3])

    counts = {}

    for idx, row in df.iterrows():
        count_words(row[0], counts)
        count_words(row[1], counts)
    
    output_path = f"wmt_counts/{year}.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, mode='w') as f:
        json.dump(counts, f)

def count_words(s: str, counts):
    for w in s.split():
        counts[w] = counts.get(w, 0) + 1
        

if __name__ == "__main__":
    main()