import pandas as pd
import base64 

dataset = pd.read_csv("data/wmt/wmt_val_full.csv")
print(len(dataset))
print(dataset.loc[(len(dataset)-1)])
