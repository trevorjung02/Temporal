import argparse
from argparse import ArgumentParser
import csv
import pandas as pd
import base64 
import random
import json
import spacy
import time
import math
import scipy.stats as ss
import numpy as np


# debug mode: does not save output files, only opens and processes 110 articles
# Implementation: Open training data as csv file with randomly chosen articles
# Process batches of sentences with spacy


def main():
    parser = ArgumentParser()
    parser.add_argument('date', type=str)
    parser.add_argument('mask_mode', type=str)
    parser.add_argument('-debug', action='store_true')
    args = parser.parse_args()
    date = args.date
    input_file = f"raw_data/wmt/news-docs.{date}.en.filtered"

    mask_chance = 0.15

    # Find total number of articles
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
        max_articles = 1000000
        train_size = 500000
        val_size = 50000
        num_load = min(max_articles, num_lines)
    print(f"Will load {num_load} articles")
    
    # Shuffle articles
    indices = list(range(num_lines))
    random.shuffle(indices)
    keep_indices = set(indices[:num_load])

    # Read articles from csv
    start = time.process_time()
    data = pd.read_csv(input_file, sep='\t', names=["date", "split"], skiprows=lambda x: x not in keep_indices, usecols=[0,1])
    print(f"Read csv: {time.process_time() - start} seconds")

    # Decode sentences from base64
    start = time.process_time()
    data['split'] = data['split'].map(lambda x: base64.b64decode(x).decode('utf-8'))
    print(f"decode sentences: {time.process_time() - start} seconds")

    # Put sentences into list
    start = time.process_time()
    sentences = []
    for row in data.itertuples():
        sentences.extend(row.split.split("\n"))
    random.shuffle(sentences)
    print(f"Create sentences: {time.process_time() - start} seconds")
    print(f"total sentences {len(sentences)}")

    # Load spacy ner
    spacy_ner = spacy.load("en_core_web_sm", disable=['tagger', 'parser','tok2vec', 'attribute_ruler', 'lemmatizer'])

    # Run spacy ner
    start = time.process_time()
    spacy_res = spacy_ner.pipe(sentences)
    print(f"Run spacy on sentences: {time.process_time() - start} seconds")

    # Mask sentences and construct dataset
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
        sentence, train_answer, val_answer = mask_sentence(sentence, args.mask_mode)
        if len(train_dataset) < train_size:
            # Add to train set
            res = [len(train_dataset), date, sentence, train_answer]
            train_dataset.append(res)
        else:
            # Add to val set
            index = len(val_dataset)
            res = [index, date, sentence, ';'.join(val_answer)]
            ids_to_answers[index] = val_answer
            val_dataset.append(res)

        
        max_input_len = max(max_input_len, len(sentence.split()))
        max_output_len = max(max_output_len, len(train_answer))
    if args.debug:
        print("___train___")
        for s in train_dataset:
            print(s)
        print("___val___")
        for s in val_dataset:
            print(s)
        print("___val answers___")
        for s in ids_to_answers:
            print(ids_to_answers[s])
    print(f"Construct datasets: {time.process_time() - start} seconds")

    print(f"max input length = {max_input_len}")
    print(f"max output length = {max_output_len}")

    if len(val_dataset) < val_size:
        print(f"Warning: train dataset has {len(train_dataset)} items, needs f{train_size} items")
        print(f"Warning: val dataset has {len(val_dataset)} items, needs f{val_size} items")

    # Write dataset
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

def mask_sentence(sentence, mode):
    if mode ==  "one_ss":
        return mask_sentence_one_ss(sentence)
    elif mode == "one_ss_random_span":
        return mask_sentence_one_ss_random_span(sentence, 3, 0.15)
    # elif mode == "mul_ss":
        # return mask_sentence_mul_ss(sentence)
    else:
        raise Exception("masking mode must be one of one_ss, one_ss_random_span, mul_ss")

def mask_sentence_one_ss(sentence):
    # i-th mask token: "<extra_id_i>"
    train_answer = []
    val_answer = []

    # Choose random named entity
    num_ents = len(sentence.ents)
    ent = sentence.ents[random.randrange(0,num_ents)]

    # Mask sentence
    text = sentence.text
    sentence = text[:ent.start_char] + "<extra_id_0>" + text[ent.end_char:]

    # Create answers
    train_answer.append(f"<extra_id_0> {ent.text} <extra_id_1")
    val_answer.append(ent.text)
    return sentence, ' '.join(train_answer), val_answer

def mask_sentence_one_ss_random_span(sentence, mean_length, mask_pct):
    # i-th mask token: "<extra_id_i>"
    train_answer = []
    val_answer = []
    span_start_indices = []
    span_start_indices_set = set()
    span_lens = []

    # Choose random named entity
    num_ents = len(sentence.ents)
    ent = sentence.ents[random.randrange(0,num_ents)]
    print(f"salient span: {ent.text}")

    # Mask salient span
    salient_span_start_char = ent.start_char
    while salient_span_start_char >= 1 and not sentence.text[salient_span_start_char-1].isspace():
        salient_span_start_char -= 1
    text = sentence.text[:salient_span_start_char].split()
    salient_span_index = len(text)
    span_start_indices.append(salient_span_index)
    span_start_indices_set.add(salient_span_index)
    span_lens.append(len(ent.text.split()))
    # sentence = text[:ent.start_char] + "<extra_id_0>" + text[ent.end_char:]

    # Sample span lengths
    text = sentence.text.split()
    print(f"Unmasked sentence: {text}")
    num_spans = math.ceil(len(text) * mask_pct / mean_length)
    # Don't do random span masking if a span will take up 25% of the text
    if len(text) <= 4 * mean_length:
        num_spans = 0
    print(f"number of spans: {num_spans}")
    if num_spans > 0:
        # span_start_indices.extend(random.sample(range(1-mean_length,len(text)), num_spans))
        span_lens.extend(random.choices(range(1, 2*mean_length), k=num_spans))
        for span_len in span_lens[1:]:
            valid = False
            while not valid:
                idx = random.randrange(1-span_len, len(text))
                valid = True
                for i in range(span_len):
                    if idx + i in span_start_indices_set:
                        valid = False
                        break
                if valid:
                    span_start_indices.append(idx)
                    span_start_indices_set.add(idx)
    
    masked_sentence = []
    print(span_start_indices)
    print(span_lens)
    sort_indices = np.argsort(span_start_indices)
    print(sort_indices)
    span_start_indices = reorder(span_start_indices, sort_indices)
    span_lens = reorder(span_lens, sort_indices)
    print(f"span_start_indices = {span_start_indices}")
    print(f"span lengths = {span_lens}")
    i = 0
    span_num = 0
    prev_index = 0
    while i < len(span_start_indices):
        j = i
        span_index = span_start_indices[j]
        span_len = span_lens[j]
        while j + 1 < len(span_start_indices):
            span_index = span_start_indices[j]
            span_len = span_lens[j]
            if span_index + span_len >= span_start_indices[j+1]:
                j += 1
            else:
                break
        span_start = max(0,span_start_indices[i])
        span_index = span_start_indices[j]
        span_end = span_index + span_len
        print(f"{span_start} {span_end}")
        masked_sentence.extend(text[prev_index:span_start])
        masked_sentence.append(f"<extra_id_{span_num}>")
        print(masked_sentence)
        masked_span = text[span_start:span_end]
        print(masked_span)
        train_answer.append(f"<extra_id_{span_num}>")
        train_answer.extend(masked_span)
        print(train_answer)
        val_answer.extend(masked_span)
        print(val_answer)
        print("-----------")
        prev_index = span_end
        i = j+1
        span_num += 1
    if prev_index < len(text):
        masked_sentence.extend(text[prev_index:])

    # Create answers
    train_answer.append(f"<extra_id_{span_num}>")
    res = ' '.join(masked_sentence), ' '.join(train_answer), [' '.join(val_answer)]
    for i in res:
        print(i)
    print("-----------")
    return res

def reorder(l, indices):
    print(l)
    print(indices)
    res = []
    for i in indices:
        res.append(l[i])
    return res

if __name__ == "__main__":
    main()