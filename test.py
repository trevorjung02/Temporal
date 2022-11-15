import pandas as pd
import numpy as np

df = pd.read_csv("data/wmt/wmt_train_full.csv")
df = np.vectorize(len)(df['input'].str.split())

# print(f"max input length = {df['input'].str.split().len().max()}")
# print(f"max output length = {df['output'].str.split().len().max()}")