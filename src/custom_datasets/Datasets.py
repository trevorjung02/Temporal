from torch.utils.data import Dataset
import pandas as pd
import json
import random

from datasets import load_dataset

class Pretrain(Dataset):
    def __init__(self, tokenizer, type_path, num_samples, input_length, output_length, args, length=None):
        self.args = args
        print(f'split is {self.args.split}')
        self.tokenizer = tokenizer
        self.type_path = type_path
        self.ssm = False
        self.dataset_version = self.args.dataset_version
        if 't5' in args.model_name_or_path:
            self.model_type='T5'
        ids_to_answers = None      
        # dataset for continual training
        if self.args.dataset == 'recentnews':
            if type_path=='train':
                if self.dataset_version=='small':
                    self.dataset = pd.read_csv('data/recent_news_small.csv')
                elif self.dataset_version=='full':
                    self.dataset = pd.read_csv('data/recent_news_full.csv')
                elif self.dataset_version=='debug':
                    self.dataset = pd.read_csv('data/recent_news_debug.csv')
            elif type_path =='split':
                if self.args.split==1:
                    if self.dataset_version=='small':
                        self.dataset = pd.read_csv('data/split/recent_news_small1.csv')
                    else:
                        raise Exception('Not supporting split for full setting.')
                elif self.args.split==2:
                    if self.dataset_version=='small':
                        self.dataset = pd.read_csv('data/split/recent_news_small2.csv')
                    else:
                        raise Exception('Not supporting split for full setting.')
                else:
                    raise Exception('Currently only supporting two splits.')
            # for mixreview pretraining corpus
            elif type_path =='pretrain':
                if self.dataset_version=='small':
                    total_line = 802776
                    skip = sorted(random.sample(range(1,total_line+1),total_line-length))
                    self.dataset = pd.read_csv('data/wikipedia_pretrain_small.csv', usecols=['input', 'output', 'original'], skiprows=skip)
                elif self.dataset_version=='full':
                    total_line = 8021155
                    skip = sorted(random.sample(range(1,total_line+1),total_line-length))
                    self.dataset = pd.read_csv('data/wikipedia_pretrain_full.csv', usecols=['input', 'output'], skiprows=skip)
        # dataset for evaluation
        else: 
            if self.args.dataset == 'templama':
                # print('Inside templama code')
                if type_path == 'train':
                    if self.args.prefix:
                        self.dataset= pd.read_csv(f'data/templama/templama_train_{self.dataset_version}_prefixed.csv')
                    else:
                        self.dataset= pd.read_csv(f'data/templama/templama_train_{self.dataset_version}.csv')
                elif type_path == 'validation':
                    if self.args.val_data is not None:
                        dataset_version = self.args.val_data
                    else:
                        dataset_version = self.dataset_version
                    if self.args.prefix:
                        self.dataset = pd.read_csv(f'data/templama/templama_val_{dataset_version}_prefixed.csv') 
                    else:
                        self.dataset = pd.read_csv(f'data/templama/templama_val_{dataset_version}.csv') 
                    with open(f'data/templama/templama_val_{dataset_version}_answers.json') as f:
                        ids_to_answers = json.load(f)  
            elif self.args.dataset == 'templama_small':
                # print('Inside templama code')
                if type_path == 'train':
                    if self.args.prefix:
                        self.dataset= pd.read_csv(f'data/templama_small_split/templama_train_{self.dataset_version}_prefixed.csv')
                    else:
                        self.dataset= pd.read_csv(f'data/templama_small_split/templama_train_{self.dataset_version}.csv')
                elif type_path == 'validation':
                    if self.args.val_data is not None:
                        dataset_version = self.args.val_data
                    else:
                        dataset_version = self.dataset_version
                    if self.args.prefix:
                        self.dataset = pd.read_csv(f'data/templama_small_split/templama_val_{dataset_version}_prefixed.csv') 
                    else:
                        self.dataset = pd.read_csv(f'data/templama_small_split/templama_val_{dataset_version}.csv') 
                    with open(f'data/templama_small_split/templama_val_{dataset_version}_answers.json') as f:
                        ids_to_answers = json.load(f)  
            elif self.args.dataset == 'wmt':
                if self.args.mask_mode == None:
                    mask_mode = ""
                else:
                    mask_mode = "_" + self.args.mask_mode
                if type_path == 'train':
                    self.dataset= pd.read_csv(f'data/wmt{mask_mode}/wmt_train_{self.dataset_version}.csv')
                elif type_path == 'validation':
                    if self.args.val_data is not None:
                        dataset_version = self.args.val_data
                    else:
                        dataset_version = self.dataset_version
                    self.dataset = pd.read_csv(f'data/wmt{mask_mode}/wmt_val_{dataset_version}.csv') 
                    with open(f'data/wmt{mask_mode}/wmt_val_{dataset_version}_answers.json') as f:
                        ids_to_answers = json.load(f) 
            elif self.args.dataset == 'streamqa':
                if type_path == 'train':
                    self.dataset= pd.read_csv(f'data/streamqa/train/{self.dataset_version}.csv')
                elif type_path == 'validation':
                    if self.args.val_data is not None:
                        dataset_version = self.args.val_data
                    else:
                        dataset_version = self.dataset_version
                    self.dataset= pd.read_csv(f'data/streamqa/val/{self.dataset_version}.csv')
                    with open(f'data/streamqa/val/{self.dataset_version}_answers.json') as f:
                        ids_to_answers = json.load(f) 
            elif self.args.dataset == 'situatedqa':
                sqa_datasets = ['2018-', '2019+', 'full']
                if type_path == 'train':
                    if not self.dataset_version in sqa_datasets:
                        raise Exception(f'Using sqa, did not provide the correct dataset version among {sqa_datasets}')
                    if self.args.prefix:
                        self.dataset= pd.read_csv(f'data/situatedqa/sqa_train_{self.dataset_version}_prefixed.csv')
                    else:
                        self.dataset= pd.read_csv(f'data/situatedqa/sqa_train_{self.dataset_version}.csv')
                elif type_path == 'validation':
                    if self.args.val_data is not None:
                        dataset_version = self.args.val_data
                    else:
                        dataset_version = self.dataset_version
                    if not dataset_version in sqa_datasets:
                        raise Exception(f'Using templama, did not provide the correct dataset version among {sqa_datasets}')
                    if self.args.prefix:
                        self.dataset = pd.read_csv(f'data/situatedqa/sqa_val_{dataset_version}_prefixed.csv') 
                    else:
                        self.dataset = pd.read_csv(f'data/situatedqa/sqa_val_{dataset_version}.csv') 
                    with open(f'data/situatedqa/sqa_val_{dataset_version}_answers.json') as f:
                        ids_to_answers = json.load(f)  
            elif self.args.dataset == 'nyt':
                if type_path == 'train':
                    self.dataset= pd.read_csv(f'data/nyt/nyt_train_{self.dataset_version}.csv')
                elif type_path == 'validation':
                    if self.args.val_data is not None:
                        dataset_version = self.args.val_data
                    else:
                        dataset_version = self.dataset_version
                    self.dataset = pd.read_csv(f'data/nyt/nyt_val_{dataset_version}.csv') 
                    with open(f'data/nyt/nyt_val_{dataset_version}_answers.json') as f:
                        ids_to_answers = json.load(f)  
            else:
                raise NameError('Select the correct Dataset!')
        print(f'Length of dataset retrieving is.. {len(self.dataset)}')
        print(self.dataset.columns)
        self.input_length = input_length
        self.output_length = output_length
        self.ids_to_answers = ids_to_answers
        self.type_path = type_path

    def __len__(self):
        return len(self.dataset)

    def convert_to_features(self, example_batch, index=None):
        year = None
        if self.args.dataset == 'recentnews':
            input_ = example_batch['input']
            target_ = example_batch['output']
            if type(input_)!=str:
                input_=''
            if type(target_)!=str:
                target_=''
        elif self.args.dataset in {'templama', 'templama_small', 'wmt','situatedqa', 'nyt', 'streamqa'}:
            input_ = example_batch['input']
            target_ = example_batch['output']
            if type(input_)!=str:
                input_=''
            if type(target_)!=str:
                target_=''
            year = example_batch['date']

        source = self.tokenizer([str(input_)], max_length=self.input_length, 
                                                    padding='max_length', truncation=True, return_tensors="pt")
        targets = self.tokenizer([str(target_)], max_length=self.output_length, 
                                                    padding='max_length', truncation=True, return_tensors="pt")     
                                                    
        if self.type_path == 'validation' or self.type_path == 'test':
            labels = example_batch['id']
        else:
            labels = None           
        ground_truth = None            
        return source, targets, labels, ground_truth, year
  
    def __getitem__(self, index):
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