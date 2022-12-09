from torch.utils.data import Dataset
import pandas as pd
import json
import random

class Preprocess(Dataset):
    def __init__(self, tokenizer, input_length, output_length, args, length=None):
        self.args = args
        self.tokenizer = tokenizer  
        self.input_length = input_length
        self.output_length = output_length

        num_lines = 0
        with open(f"news-docs.{self.args.date}.en.filtered", 'r') as fp:
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

        self.dataset = pd.read_csv(f"news-docs.{self.args.date}.en.filtered", sep='\t', names=["date", "split"], skiprows=lambda x: x not in keep_indices, usecols=[0,1])

        print(f'Length of dataset retrieving is.. {len(self.dataset)}')
        print(self.dataset.columns)

    def __len__(self):
        return len(self.dataset)

    def convert_to_features(self, example_batch, index=None):
        input_ = example_batch['input']
        if type(input_)!=str:
            input_=''
        if type(target_)!=str:
            target_=''
        year = example_batch['date']
        source = self.tokenizer.batch_encode_plus([str(input_)], max_length=self.input_length, 
                                                    padding='max_length', truncation=True, return_tensors="pt")
        targets = self.tokenizer.batch_encode_plus([str(target_)], max_length=self.output_length, 
                                                    padding='max_length', truncation=True, return_tensors="pt")     
        if self.type_path == 'validation' and self.model_type =='GPT2':
            ground_truth = self.tokenizer.batch_encode_plus([str(ground_truth_)], max_length=self.output_length, 
                                                    padding='max_length', truncation=True, return_tensors="pt")  
        else: 
            ground_truth = None
        if (self.args.dataset == 'invariantlama' or self.args.dataset== 'TriviaQA' or self.args.dataset== 'fever' or self.args.dataset== 'AY2' or self.args.dataset== 'WNED' or self.args.dataset== 'CWEB' 
        or self.args.dataset== 'TREX' or self.args.dataset== 'zsRE' or self.args.dataset== 'NQ' or self.args.dataset== 'HotpotQA' or self.args.dataset== 'ELI5' or self.args.dataset== 'WOW' or (self.args.dataset in {'templama', 'templama_small', 'situatedqa', 'wmt', 'streamqa'} and self.type_path in {'validation', 'test'})):
            labels = example_batch['id']
        elif (self.args.dataset == 'newlama' or self.args.dataset == 'updatedlama' or self.args.dataset == 'newlama_easy' or self.args.dataset == 'newqa_easy'):
            labels = example_batch['unique_id']
        else:
            labels = None                       
        return source, targets, labels, ground_truth, year
  
    def __getitem__(self, index):
        if (self.args.dataset== 'TriviaQA' or self.args.dataset== 'fever' or self.args.dataset== 'AY2' or self.args.dataset== 'WNED' or self.args.dataset== 'CWEB' 
        or self.args.dataset== 'TREX' or self.args.dataset== 'zsRE' or self.args.dataset== 'NQ' or self.args.dataset== 'HotpotQA' or self.args.dataset== 'ELI5' or self.args.dataset== 'WOW'):
            source, targets, labels, ground_truth, year = self.convert_to_features(self.dataset[index])
        else:
            source, targets, labels, ground_truth, year = self.convert_to_features(self.dataset.iloc[index])
        
        source_ids = source["input_ids"].squeeze()
        target_ids = targets["input_ids"].squeeze()

        src_mask    = source["attention_mask"].squeeze()
        target_mask = targets["attention_mask"].squeeze()

        if labels is not None:
            label_ids = labels
        else:
            label_ids = -1
        
        if ground_truth is not None:
            ground_truth_ids = ground_truth["input_ids"].squeeze()
        else: 
            ground_truth_ids = -1

        return {"source_ids": source_ids, "source_mask": src_mask, "target_ids": target_ids, "target_mask": target_mask, "label_ids": label_ids, "ground_truth_ids": ground_truth_ids, "year": year}