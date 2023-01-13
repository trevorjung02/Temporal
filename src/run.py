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
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.profilers import PyTorchProfiler
from models import load_model
import wandb
import copy
from adapter_utils.extract_adapters import extract_adapters

from custom_datasets.Datasets import Pretrain
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

def main():
    cli_args = get_cli_args()
    hparams = get_hparams(cli_args.config)
    args = get_args(hparams, cli_args)

    if hparams.dataset == 'wmt':
        val_check_interval = 5000
    else:
        val_check_interval = 1.0
    args['val_check_interval'] = val_check_interval

    #Logging into WANDB if needed
    if hparams.wandb_log:
        wandb.init(project=hparams.wandb_project, name=f"{hparams.method}_{args['dataset_version']}" , config=args, settings=wandb.Settings(start_method="fork"))
        wandb_logger = WandbLogger(project=hparams.wandb_project)
        wandb.define_metric("em_score", summary="max")
        wandb.define_metric("f1_score", summary="max")
        wandb.define_metric("loss")

        args = wandb.config.as_dict()
    else:
        wandb_logger = None 

    args = argparse.Namespace(**args)
    callbacks = get_callbacks(args)

    get_output_path(args, cli_args)

    if args.method == 'kadapter_ensemble':
        args.years_to_paths = {}
        for year in range(args.year_start, args.year_end+1):
            dir = f"outputs/wmtkadapter_{year}_2freeze_{''.join(map(str, args.adapter_list))}_{args.adapter_hidden_size}"
            args.years_to_paths[str(year)] = extract_adapters(dir)

    args.adapter_config = {'adapter_list': args.adapter_list, 'adapter_hidden_size': args.adapter_hidden_size, 'adapter_enc_dec': args.adapter_enc_dec, 'pool_size': args.pool_size, 'years_to_paths': args.years_to_paths, 'load_adapters': args.load_adapters}
    
    print(args)

    # Logging Learning Rate Scheduling
    if args.use_lr_scheduling and hparams.wandb_log:
        callbacks.append(pl.callbacks.LearningRateMonitor())
        
    train_params = dict(
        devices=args.n_gpu,
        max_epochs=args.num_train_epochs,
        precision= args.precision,
        gradient_clip_val=args.max_grad_norm,
        val_check_interval=args.val_check_interval,
        logger=wandb_logger,
        callbacks = callbacks,
        accelerator=args.accelerator,
        strategy=args.strategy,
        auto_lr_find=args.find_lr,
        profiler = PyTorchProfiler(filename='perf-log') if args.use_profiler else None
    )
    if args.strategy == 'ddp':
        train_params['strategy'] = DDPStrategy(find_unused_parameters=False)

    #Getting the Model type & Method
    if 't5' in args.model_name_or_path:
        model_type='T5'
    elif 'gpt2' in args.model_name_or_path:
        model_type='GPT2'
    else:
        raise Exception('Select the correct model. Supporting "t5" and "gpt2" only.')
    Model: pl.LightningModule = load_model(type=model_type)
    model: pl.LightningModule = Model(args)

    trainer = pl.Trainer(**train_params)
    
    torch.cuda.empty_cache()
    print(f"---------------torch memory allocated: {torch.cuda.memory_allocated()}---------------")

    if args.check_validation_only:
        trainer.validate(model)
    else:
        if args.find_lr:
            trainer.tune(model)
            wandb.config.update({'learning_rate': model.hparams.learning_rate}, allow_val_change=True)
            wandb.update()
        if args.resume_from_checkpoint:
            trainer.fit(model, ckpt_path=args.checkpoint_path)
        else:
            if args.checkpoint_path:
                model = Model.load_from_checkpoint(args.checkpoint_path, hparams=args)
            trainer.fit(model)

def get_cli_args():
    parser = ArgumentParser()
    parser.add_argument('--config', default=None, type=str)
    parser.add_argument('-datav', type=str)
    parser.add_argument('-val_data', type=str)
    parser.add_argument('-checkpoint_path', type=str)
    parser.add_argument('-find_lr', action='store_true')
    parser.add_argument('-lr', type=float)
    parser.add_argument('-debug', action='store_true')
    arg_ = parser.parse_args()
    if arg_.config == None:
        raise NameError("Include a config file in the argument please.")
    return arg_

def get_hparams(config):
    #Getting configurations
    with open(config) as config_file:
        hparams = json.load(config_file)
    hparams = argparse.Namespace(**hparams)

    #Init configs that are not given
    if 'split_num' not in hparams:
        hparams.split_num = 1
    if 'split' not in hparams:
        hparams.split = 0
    if 'grad_norm' not in hparams:
        hparams.grad_norm = 0.5
    if 'weight_decay' not in hparams:
        hparams.weight_decay = 0.0
    if 'output_log' not in hparams:
        hparams.output_log = None
    if 'prefix' not in hparams:
        hparams.prefix = None
    if 't5_learning_rate' not in hparams:
        hparams.t5_learning_rate = None
    if 'checkpoint_dir' not in hparams:
        hparams.checkpoint_dir = None
    if 'adapter_enc_dec' not in hparams:
        hparams.adapter_enc_dec = None
    if 'adapter_list' not in hparams:
        hparams.adapter_list = None
    if 'adapter_hidden_size' not in hparams:
        hparams.adapter_hidden_size = None
    if 'pool_size' not in hparams:
        hparams.pool_size = None
    if 'val_data' not in hparams:
        hparams.val_data = None
    if 'years_to_paths' not in hparams:
        hparams.years_to_paths = None
    if 'load_adapters' not in hparams:
        hparams.load_adapters = None
    if 'year_start' not in hparams:
        hparams.year_start = None
    if 'year_end' not in hparams:
        hparams.year_end = None
    if 'mask_mode' not in hparams:
        hparams.mask_mode = None
    if 'strategy' not in hparams:
        hparams.strategy = None
    if 'precision' not in hparams:
        hparams.precision = 32
    if 'accelerator' not in hparams:
        hparams.accelerator = 'gpu'
    if 'use_profiler' not in hparams:
        hparams.use_profiler = False
    return hparams

def get_args(hparams, cli_args):
    #Setting configurations
    args = dict(
        output_dir=hparams.output_dir, # Path to save the checkpoints
        dataset=hparams.dataset,
        dataset_version = hparams.dataset_version,
        prefix = hparams.prefix,
        split_num = hparams.split_num,
        split = hparams.split,
        model_name_or_path=hparams.model,
        method=hparams.method,
        freeze_level=hparams.freeze_level,
        mode=hparams.mode,
        tokenizer_name_or_path=hparams.model,
        max_input_length=hparams.input_length,
        max_output_length=hparams.output_length,
        freeze_encoder=False,
        freeze_embeds=False,
        learning_rate=hparams.learning_rate,
        weight_decay=hparams.weight_decay,
        adam_epsilon=1e-8,
        warmup_steps=0,
        train_batch_size=hparams.train_batch_size,
        eval_batch_size=hparams.train_batch_size,
        num_train_epochs=hparams.num_train_epochs,
        n_gpu=hparams.ngpu,
        num_workers=hparams.num_workers,
        resume_from_checkpoint=hparams.resume_from_checkpoint, 
        use_lr_scheduling = hparams.use_lr_scheduling,
        n_val=-1,
        n_train=-1,
        n_test=-1,
        early_stop_callback=False,
        opt_level='O1', # you can find out more on optimisation levels here https://nvidia.github.io/apex/amp.html#opt-levels-and-properties
        max_grad_norm=hparams.grad_norm, # if you enable 16-bit training then set this to a sensible value, 0.5 is a good default
        seed=42,
        check_validation_only=hparams.check_validation,
        checkpoint_path=hparams.checkpoint_path,
        accelerator=hparams.accelerator,
        output_log=hparams.output_log,
        wandb_log = hparams.wandb_log,
        adapter_list = hparams.adapter_list,
        adapter_hidden_size = hparams.adapter_hidden_size,
        t5_learning_rate = hparams.t5_learning_rate,
        checkpoint_dir = hparams.checkpoint_dir,
        adapter_enc_dec = hparams.adapter_enc_dec,
        pool_size = hparams.pool_size,
        val_data = hparams.val_data,
        years_to_paths = hparams.years_to_paths,
        load_adapters = hparams.load_adapters,
        year_start = hparams.year_start,
        year_end = hparams.year_end,
        mask_mode=hparams.mask_mode,
        strategy=hparams.strategy,
        precision=hparams.precision,
        use_profiler=hparams.use_profiler
    )
    update_args_with_cli(args, cli_args)
    return args

def update_args_with_cli(args, cli_args):
    if cli_args.datav is not None:
        args['dataset_version'] = cli_args.datav
    if cli_args.val_data is not None:
        args['val_data'] = cli_args.val_data
    else:
        args['val_data'] = args['dataset_version']
    args['find_lr'] = cli_args.find_lr
    args['debug'] = cli_args.debug
    if cli_args.lr:
        args['learning_rate'] = cli_args.lr

def get_callbacks(args):
    if args.output_dir=="":
        callbacks=[]
    else:
        args.output_dir += args.method
        args.output_dir += '_' + str(args.dataset_version)
        if 'kadapter' in args.method: 
            args.output_dir += '_' + str(args.freeze_level) + 'freeze'
            args.output_dir += '_' + ''.join(map(str, args.adapter_list))
            args.output_dir += '_' + str(args.adapter_hidden_size)
            if args.adapter_enc_dec:
                args.output_dir += '_' + 'encdec'
        if args.dataset == 'wmt':
            # if args.dataset_version == 'full':
            #     # _________________________________Debug_____________
            #     every_n_train_steps=10
            #     # every_n_train_steps=2500
            # else:
            #     # _________________________________Debug_____________
            #     every_n_train_steps=10
            if 't5' in args.model_name_or_path:
                monitor="f1_score"
                mode="max"
            elif 'gpt2' in args.model_name_or_path:
                monitor="loss"
                mode="min"
            every_n_train_steps=2500
            callbacks = [ModelCheckpoint(dirpath = args.output_dir, filename = '{epoch}-{f1_score:.4f}-{em_score:.4f}', save_top_k=1, every_n_train_steps=every_n_train_steps, mode=mode, monitor=monitor, save_last=True)]
        elif args.dataset == 'nyt':
            if args.dataset_version == 'full':
                every_n_train_steps=2500
                callbacks = [ModelCheckpoint(dirpath = args.output_dir, filename = '{epoch}-{f1_score:.4f}-{em_score:.4f}', save_top_k=2, every_n_train_steps=every_n_train_steps, mode="max", monitor="f1_score")]
            else:
                callbacks = [ModelCheckpoint(dirpath = args.output_dir, filename = '{epoch}-{f1_score:.4f}-{em_score:.4f}', save_top_k=2, every_n_epochs=1, mode="max", monitor="f1_score")]
        elif args.dataset_version == 'debug':
            every_n_train_steps=1
            callbacks = [ModelCheckpoint(dirpath = args.output_dir, filename = '{epoch}-{f1_score:.4f}-{em_score:.4f}', save_top_k=2, every_n_train_steps=every_n_train_steps, mode="max", monitor="f1_score")]
        elif args.dataset == 'templama':
            callbacks = [ModelCheckpoint(dirpath = args.output_dir, filename = '{epoch}-{f1_score:.4f}-{em_score:.4f}', save_top_k=2, every_n_train_steps=100, mode="max", monitor="em_score")]
        else:
            callbacks = [ModelCheckpoint(dirpath = args.output_dir, filename = '{epoch}-{f1_score:.4f}-{em_score:.4f}', save_top_k=2, every_n_epochs=1, mode="max", monitor="em_score")]
    return callbacks

def get_output_path(args, cli_args):
    if args.checkpoint_dir is not None:
        pattern = 'em_score=(\d.\d*)'
        checkpoint_path = None
        max_em = 0
        for filename in os.listdir(args.checkpoint_dir):
            f = os.path.join(args.checkpoint_dir, filename)
            # checking if it is a file
            if os.path.isfile(f) and os.path.splitext(f)[1] == '.ckpt':
                em = float(re.search(pattern, filename).group(1))
                if em > max_em:
                    max_em = em
                    checkpoint_path = f
        args.checkpoint_path = checkpoint_path
        print(f"checkpoint path = {args.checkpoint_path}")

    if cli_args.checkpoint_path is not None:
        args.checkpoint_path = cli_args.checkpoint_path

if __name__ == '__main__':
    main()