import argparse
from argparse import ArgumentParser
import csv
import pandas as pd
import base64 
import random
import json

parser = ArgumentParser()
parser.add_argument('date', type=str)
arg_ = parser.parse_args()
date = arg_.date

with open(f"data/wmt/wmt_val_{date}_answers.json", "r", encoding='utf-8') as f:
    ids_to_answers = json.load(f)

sentinel_len = len("<extra_id_0> ")
for id in ids_to_answers:
    ids_to_answers[id] = [ids_to_answers[id][sentinel_len:] + ids_to_answers[id][:-sentinel_len]]

with open(f"data/wmt/wmt_val_{date}_answers.json", "w", encoding='utf-8') as f:
    json.dump(ids_to_answers, f, ensure_ascii=False)