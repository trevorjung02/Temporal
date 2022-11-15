import pandas as pd
import csv
import json

def main():
    num_data = 10000
    for year in range(2019,2021):
        data = pd.read_csv(f"data/wmt_large/wmt_val_{year}.csv", nrows=num_data)
        data.to_csv(f"data/wmt/wmt_val_{year}.csv", index=False)
        
        mask_len = len("<extra_id_0> ")
        with open(f"data/wmt_large/wmt_val_{year}_answers.json", encoding='utf-8') as f:
            answers = json.load(f)
            answers = {str(int(k)):[v[0][mask_len:-mask_len]] for k,v in answers.items() if int(k) < num_data}
        with open(f"data/wmt/wmt_val_{year}_answers.json", encoding='utf-8', mode='w') as f:
            json.dump(answers,f)

if __name__ == "__main__":
    main()