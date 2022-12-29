import argparse
import os
import pandas as pd
import json

def main():
    for year in range(2008, 2021):
        build_sentence(year)
    
def build_sentence(year):
    path = f"data/wmt/wmt_train_{year}.csv"
    df = pd.read_csv(path, usecols=[2,3])
    
    sentences = []
    for idx, row in df.iterrows():
        sentences.append(row[0])
        sentences.append(row[1])
    s = ' '.join(sentences)
    
    output_path = f"wmt_sentences/{year}.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, mode='w') as f:
        json.dump(s, f)

if __name__ == "__main__":
    main()