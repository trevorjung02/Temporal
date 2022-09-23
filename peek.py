import pandas as pd
import base64 

dataset = pd.read_csv("data/wmt/wmt_val_full.csv", nrows=100)
print(dataset.columns)
print(dataset)
print(dataset['input'])

