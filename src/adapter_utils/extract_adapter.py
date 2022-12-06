import argparse
from argparse import ArgumentParser
import os
import re
import json
import random
from scripts.evaluation import evaluate
import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from models import load_model
import wandb
import copy

from custom_datasets.Datasets import Pretrain
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

def main():
    parser = ArgumentParser()
    parser.add_argument('checkpoint_path', type=str)
    args = parser.parse_args()

    output_path = args.checkpoint_path.replace('.ckpt', '_AdapterWeights.ckpt')
    checkpoint = torch.load(args.checkpoint_path)
    state_dict = checkpoint['state_dict']

    for key in list(state_dict.keys()):
        if 'kadapter' not in key:
            del state_dict[key]
        else:
            # new_key = key.replace('enc_kadapter.', '')
            new_key = key.replace('model.enc_kadapter.', '')
            state_dict[new_key] = state_dict.pop(key)
    
    print(state_dict.keys())

    torch.save(state_dict, output_path)

if __name__ == '__main__':
    main()