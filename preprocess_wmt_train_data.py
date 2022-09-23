import argparse
from argparse import ArgumentParser
import csv
import pandas as pd
import base64 
import random
import json
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
import spacy
import torch

# debug mode: does not save output files, only opens and processes 110 articles

def main():
    parser = ArgumentParser()
    parser.add_argument('date', type=str)
    parser.add_argument('-debug', action='store_true')
    args = parser.parse_args()
    date = args.date

    mask_chance = 0.15

    num_lines = 0
    with open(f"news-docs.{date}.en.filtered", 'r') as fp:
        for count, line in enumerate(fp):
            pass
        num_lines = count + 1

    # train_size, dev_size: number of articles in train and dev sets
    if args.debug:
        train_size = 100
        dev_size = 10
    else:
        train_size = 20000
        dev_size = 1000
    indices = list(range(num_lines))
    random.shuffle(indices)
    keep_indices = set(indices[:train_size + dev_size])

    data = pd.read_csv(f"news-docs.{date}.en.filtered", sep='\t', names=["date", "split"], skiprows=lambda x: x not in keep_indices, usecols=[0,1])

    tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
    model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")

    bert_ner = pipeline("ner", model=model, tokenizer=tokenizer, device=0)
    spacy_ner = spacy.load("en_core_web_sm")

    data['split'] = data['split'].map(lambda x: base64.b64decode(x).decode('utf-8'))

    train_dataset = []
    val_dataset = []
    ids_to_answers = {}
    max_input_len = 0
    max_output_len = 0
    train_index = 0
    val_index = 0 
    row_number = 0
    for row in data.itertuples():
        if row_number % 1000 == 0:
            print(row_number)
        article = row.split
        sentences = article.split('\n')
        batch_size = 128
        batch_num = 0 
        while batch_num * batch_size < len(sentences):
            bert_res = bert_ner(sentences[batch_num*batch_size:(batch_num+1)*batch_size])
            spacy_res = list(spacy_ner.pipe(sentences[batch_num*batch_size:(batch_num+1)*batch_size]))
            for i in range(batch_num*batch_size, min(len(sentences), (batch_num+1)*batch_size)):
                words = sentences[i].split()
                if len(words) == 0:
                    continue
                sentence, answers = mask_sentence(sentences[i], mask_chance, bert_res[i % batch_size], spacy_res[i % batch_size])
                if row_number < train_size:
                    res = [train_index, date, sentence, ' '.join(answers)]
                    train_dataset.append(res)
                    train_index += 1
                else:
                    res = [val_index, date, sentence, ' '.join(answers)]
                    val_dataset.append(res)
                    ids_to_answers[str(val_index)] = [' '.join(answers)]
                    val_index += 1
                
                max_input_len = max(max_input_len, len(words))
                max_output_len = max(max_output_len, len(answers))
            batch_num += 1
        row_number +=1 
    if args.debug:
        print(train_dataset)
        print(val_dataset)

    print(f"max input length = {max_input_len}")
    print(f"max output length = {max_output_len}")

    if not args.debug:
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

def mask_sentence(sentence, mask_chance, bert_res, spacy_res):
    # print(sentence)
    combined = []
    for i in range(len(bert_res)):
        # print(bert_res[i])
        if bert_res[i]['entity'][0] != 'B' and len(combined) > 0 and bert_res[i]['start'] - combined[-1]['end'] < 2:
            combined[-1]['end'] = bert_res[i]['end']
        else:
            combined.append(bert_res[i])
    bert_res = combined

    dates = [ent for ent in spacy_res.ents if ent.label_ == 'DATE']

    num_ents = len(bert_res) + len(dates)
    if num_ents > 0:
        i = random.randrange(0, num_ents)
        mask_token = "<extra_id_0>"
        if i < len(bert_res):
            ent = bert_res[i]
            while ent['start']-1 >= 0 and sentence[ent['start']-1].isalpha():
                ent['start'] -= 1
            while ent['end'] < len(sentence) and sentence[ent['end']].isalpha():
                ent['end'] += 1
            answers = [f"{mask_token} {sentence[ent['start']: ent['end']]}"]
            sentence = sentence[:ent['start']] + mask_token + sentence[ent['end']:]
        else:
            ent = dates[i - len(bert_res)]
            sentence = sentence[:ent.start_char] + mask_token + sentence[ent.end_char:]
            answers = [f"{mask_token} {ent.text}"]
    else:
        words = sentence.split()
        answers = []
        done = False
        prev_masked = False
        while not done:
            for j in range(len(words)):
                if random.random() < mask_chance:
                    if prev_masked:
                        answers[-1] += f' {words[j]}'
                        words[j] = None
                    else:
                        mask_token = f"<extra_id_{len(answers)}>"
                        answers.append(f"{mask_token} {words[j]}")
                        words[j] = mask_token
                    done = True
                    prev_masked = True
                else:
                    prev_masked = False 
        sentence = ' '.join([w for w in words if w is not None])
    answers.append(f"<extra_id_{len(answers)}>")
    return sentence, answers

if __name__ == "__main__":
    main()