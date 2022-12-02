import argparse
from argparse import ArgumentParser
import csv
import pandas as pd
import base64 
import random
import json
import spacy
import time

# debug mode: does not save output files, only opens and processes 110 articles
# Implementation: Open training data as csv file with randomly chosen articles
# Process batches of sentences with spacy


def main():
    parser = ArgumentParser()
    parser.add_argument('date', type=str)
    parser.add_argument('-debug', action='store_true')
    args = parser.parse_args()
    date = args.date
    input_file = f"news-docs.{date}.en.filtered"

    mask_chance = 0.15

    num_lines = 0
    with open(input_file, 'r') as fp:
        for count, line in enumerate(fp):
            pass
        num_lines = count + 1
    print(f"number of articles: {num_lines}")

    # train_size, val_size: number of articles in train and dev sets
    if args.debug:
        train_size = 100
        val_size = 10
        num_load = 100
    else:
        train_size = 500000
        val_size = 50000
        num_load = min(300000, num_lines)
    print(f"{num_load} articles loaded")
    indices = list(range(num_lines))
    random.shuffle(indices)
    keep_indices = set(indices[:num_load])

    start = time.process_time()
    data = pd.read_csv(input_file, sep='\t', names=["date", "split"], skiprows=lambda x: x not in keep_indices, usecols=[0,1])
    print(f"Read csv: {time.process_time() - start} seconds")

    start = time.process_time()
    data['split'] = data['split'].map(lambda x: base64.b64decode(x).decode('utf-8'))
    print(f"decode sentences: {time.process_time() - start} seconds")

    start = time.process_time()
    sentences = []
    for row in data.itertuples():
        sentences.extend(row.split.split("\n"))
    print(f"Create sentences: {time.process_time() - start} seconds")
    print(f"total sentences {len(sentences)}")

    spacy_ner = spacy.load("en_core_web_sm", disable=['tagger', 'parser','tok2vec', 'attribute_ruler', 'lemmatizer'])

    start = time.process_time()
    spacy_res = spacy_ner.pipe(sentences)
    print(f"Run spacy on sentences: {time.process_time() - start} seconds")

    start = time.process_time()
    train_dataset = []
    val_dataset = []
    ids_to_answers = {}
    max_input_len = 0
    max_output_len = 0
    for sentence in spacy_res:
        if len(sentence.ents) == 0:
            continue
        if len(val_dataset) >= val_size:
            break
        sentence, answers = mask_sentence(sentence)
        if len(train_dataset) < train_size:
            res = [len(train_dataset), date, sentence, ' '.join(answers)]
            train_dataset.append(res)
        else:
            index = len(val_dataset)
            res = [index, date, sentence, ' '.join(answers)]
            ids_to_answers[index] = [' '.join(answers)]
            val_dataset.append(res)

        
        max_input_len = max(max_input_len, len(sentence.split()))
        max_output_len = max(max_output_len, len(answers))
    if args.debug:
        for s in train_dataset:
            print(s)
        for s in val_dataset:
            print(s)
    print(f"Construct datasets: {time.process_time() - start} seconds")

    print(f"max input length = {max_input_len}")
    print(f"max output length = {max_output_len}")

    if len(val_dataset) < val_size:
        print(f"Warning: train dataset has {len(train_dataset)} items, needs f{train_size} items")
        print(f"Warning: val dataset has {len(val_dataset)} items, needs f{val_size} items")

    if not args.debug:
        start = time.process_time()
        with open(f"data/wmt/wmt_train_{date}.csv", "w", encoding='utf-8') as csvfile:
            w = csv.writer(csvfile)
            w.writerow(["id", "date", "input", "output"])
            w.writerows(train_dataset)

        with open(f"data/wmt/wmt_val_{date}.csv", "w", encoding='utf-8') as csvfile:
            w = csv.writer(csvfile)
            w.writerow(["id", "date", "input", "output"])
            w.writerows(val_dataset)

        with open(f"data/wmt/wmt_val_{date}_answers.json", "w", encoding='utf-8') as f:
            json.dump(ids_to_answers, f, ensure_ascii=False)
        print(f"Write datasets: {time.process_time() - start} seconds")

def mask_sentence(sentence):
    mask_token = "<extra_id_0>"
    num_ents = len(sentence.ents)
    ent = sentence.ents[random.randrange(0,num_ents)]
    text = sentence.text
    sentence = text[:ent.start_char] + mask_token + text[ent.end_char:]
    answers = [f"{mask_token} {ent.text}"]
    answers.append(f"<extra_id_{len(answers)}>")
    return sentence, answers

if __name__ == "__main__":
    main()