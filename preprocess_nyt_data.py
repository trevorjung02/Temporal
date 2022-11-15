from argparse import ArgumentParser
import csv
import random
import json
import spacy
import time

# debug mode: does not save output files, only opens and processes 110 articles
# Implementation: 
# Open training data with randomly chosen articles
# Process batches of sentences with spacy


def main():
    parser = ArgumentParser()
    parser.add_argument('start_date', type=int)
    parser.add_argument('-end_date', type=int)
    parser.add_argument('-debug', action='store_true')
    args = parser.parse_args()
    if args.end_date:
        for date in range(args.start_date, args.end_date+1):
            process_file(date, args)
    else:
        process_file(args.start_date, args)

def process_file(date, args):
    input_file = f"nyt/nyt_{date}.txt"
    mask_chance = 0.15

    num_lines = 0
    with open(input_file, 'r', encoding='utf-8') as fp:
        for count, line in enumerate(fp):
            pass
        num_lines = count + 1
    print(f"number of sentences: {num_lines}")

    # train_size, val_size: number of sentences in train and dev sets
    if args.debug:
        train_size = 100
        val_size = 10
    else:
        train_size = 9000
        val_size = 1000
    num_load = min(train_size + val_size, num_lines)
    print(f"{num_load} sentences loaded")
    indices = list(range(num_lines))
    random.shuffle(indices)
    keep_indices = set(indices[:num_load])

    start = time.process_time()
    with open(input_file, encoding='utf-8') as f:
        sentences = f.read().splitlines()
    sentences = [sentences[i] for i in keep_indices]
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
        sentence, masked_answers, answers = mask_sentence(sentence, mask_chance)
        if len(train_dataset) < train_size:
            res = [len(train_dataset), date, sentence, masked_answers]
            train_dataset.append(res)
        else:
            index = len(val_dataset)
            res = [index, date, sentence, masked_answers]
            ids_to_answers[index] = answers
            val_dataset.append(res)

        max_input_len = max(max_input_len, len(sentence.split()))
        max_output_len = max(max_output_len, len(masked_answers.split()))

    if args.debug:
        print(f"train dataset has {len(train_dataset)} lines")
        print(f"val dataset has {len(val_dataset)} lines")
        for s in train_dataset:
            print(s)
        for s in val_dataset:
            print(s)
        print(ids_to_answers)
    print(f"Construct datasets: {time.process_time() - start} seconds")

    print(f"max input length = {max_input_len}")
    print(f"max output length = {max_output_len}")

    if len(val_dataset) < val_size:
        print(f"Warning: train dataset has {len(train_dataset)} items, needs {train_size} items")
        print(f"Warning: val dataset has {len(val_dataset)} items, needs {val_size} items")

    if not args.debug:
        start = time.process_time()
        with open(f"data/nyt/nyt_train_{date}.csv", "w", encoding='utf-8') as csvfile:
            w = csv.writer(csvfile)
            w.writerow(["id", "date", "input", "output"])
            w.writerows(train_dataset)

        with open(f"data/nyt/nyt_val_{date}.csv", "w", encoding='utf-8') as csvfile:
            w = csv.writer(csvfile)
            w.writerow(["id", "date", "input", "output"])
            w.writerows(val_dataset)

        with open(f"data/nyt/nyt_val_{date}_answers.json", "w", encoding='utf-8') as f:
            json.dump(ids_to_answers, f, ensure_ascii=False)
        print(f"Write datasets: {time.process_time() - start} seconds")

def mask_sentence(sentence, mask_chance):
    mask_token = "<extra_id_0>"
    num_ents = len(sentence.ents)
    text = sentence.text
    if num_ents > 0:
        ent = sentence.ents[random.randrange(0,num_ents)]
        sentence = text[:ent.start_char] + mask_token + text[ent.end_char:]
        masked_answers = [f"{mask_token} {ent.text}"]
        answers = [ent.text]
    else:
        words = text.split()
        masked_answers = []
        answers = []
        done = False
        prev_masked = False
        while not done:
            for j in range(len(words)):
                if random.random() < mask_chance:
                    if prev_masked:
                        masked_answers[-1] += f' {words[j]}'
                        answers[-1] += f' {words[j]}'
                        words[j] = None
                    else:
                        mask_token = f"<extra_id_{len(masked_answers)}>"
                        masked_answers.append(f"{mask_token} {words[j]}")
                        answers.append(words[j])
                        words[j] = mask_token
                    done = True
                    prev_masked = True
                else:
                    prev_masked = False 
        sentence = ' '.join([w for w in words if w is not None])
    masked_answers.append(f"<extra_id_{len(answers)}>")
    return sentence, ' '.join(masked_answers), answers

if __name__ == "__main__":
    main()