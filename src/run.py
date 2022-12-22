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
from adapter_utils.extract_adapters import extract_adapters

from custom_datasets.Datasets import Pretrain
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

def main():
    parser = ArgumentParser()
    parser.add_argument('--config', default=None, type=str)
    parser.add_argument('-datav', type=str)
    parser.add_argument('-val_data', type=str)
    parser.add_argument('-checkpoint_path', type=str)
    parser.add_argument('-find_lr', action='store_true')
    parser.add_argument('-lr', type=float)
    arg_ = parser.parse_args()
    if arg_.config == None:
        raise NameError("Include a config file in the argument please.")

    #Getting configurations
    with open(arg_.config) as config_file:
        hparam = json.load(config_file)
    hparam = argparse.Namespace(**hparam)

    #Setting GPUs to use
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=hparam.CUDA_VISIBLE_DEVICES

    #Init configs that are not given
    if 'split_num' not in hparam:
        hparam.split_num = 1
    if 'split' not in hparam:
        hparam.split = 0
    if 'grad_norm' not in hparam:
        hparam.grad_norm = 0.5
    if 'weight_decay' not in hparam:
        hparam.weight_decay = 0.0
    if 'output_log' not in hparam:
        hparam.output_log = None
    if 'prefix' not in hparam:
        hparam.prefix = None
    if 't5_learning_rate' not in hparam:
        hparam.t5_learning_rate = None
    if 'checkpoint_dir' not in hparam:
        hparam.checkpoint_dir = None
    if 'adapter_enc_dec' not in hparam:
        hparam.adapter_enc_dec = None
    if 'adapter_list' not in hparam:
        hparam.adapter_list = None
    if 'adapter_hidden_size' not in hparam:
        hparam.adapter_hidden_size = None
    if 'pool_size' not in hparam:
        hparam.pool_size = None
    if 'val_data' not in hparam:
        hparam.val_data = None
    if 'years_to_paths' not in hparam:
        hparam.years_to_paths = None
    if 'load_adapters' not in hparam:
        hparam.load_adapters = None
    if 'year_start' not in hparam:
        hparam.year_start = None
    if 'year_end' not in hparam:
        hparam.year_end = None
    if 'mask_mode' not in hparam:
        hparam.mask_mode = None

    #Setting configurations
    args = dict(
        output_dir=hparam.output_dir, # Path to save the checkpoints
        dataset=hparam.dataset,
        dataset_version = hparam.dataset_version,
        prefix = hparam.prefix,
        split_num = hparam.split_num,
        split = hparam.split,
        model_name_or_path=hparam.model,
        method=hparam.method,
        freeze_level=hparam.freeze_level,
        mode=hparam.mode,
        tokenizer_name_or_path=hparam.model,
        max_input_length=hparam.input_length,
        max_output_length=hparam.output_length,
        freeze_encoder=False,
        freeze_embeds=False,
        learning_rate=hparam.learning_rate,
        weight_decay=hparam.weight_decay,
        adam_epsilon=1e-8,
        warmup_steps=0,
        train_batch_size=hparam.train_batch_size,
        eval_batch_size=hparam.train_batch_size,
        num_train_epochs=hparam.num_train_epochs,
        n_gpu=hparam.ngpu,
        num_workers=hparam.num_workers,
        resume_from_checkpoint=hparam.resume_from_checkpoint, 
        use_lr_scheduling = hparam.use_lr_scheduling,
        n_val=-1,
        n_train=-1,
        n_test=-1,
        early_stop_callback=False,
        use_deepspeed=hparam.use_deepspeed,
        opt_level='O1', # you can find out more on optimisation levels here https://nvidia.github.io/apex/amp.html#opt-levels-and-properties
        max_grad_norm=hparam.grad_norm, # if you enable 16-bit training then set this to a sensible value, 0.5 is a good default
        seed=42,
        check_validation_only=hparam.check_validation,
        checkpoint_path=hparam.checkpoint_path,
        accelerator=hparam.accelerator,
        output_log=hparam.output_log,
        wandb_log = hparam.wandb_log,
        adapter_list = hparam.adapter_list,
        adapter_hidden_size = hparam.adapter_hidden_size,
        t5_learning_rate = hparam.t5_learning_rate,
        checkpoint_dir = hparam.checkpoint_dir,
        adapter_enc_dec = hparam.adapter_enc_dec,
        pool_size = hparam.pool_size,
        val_data = hparam.val_data,
        years_to_paths = hparam.years_to_paths,
        load_adapters = hparam.load_adapters,
        year_start = hparam.year_start,
        year_end = hparam.year_end,
        mask_mode=hparam.mask_mode
    )
    if arg_.datav is not None:
        args['dataset_version'] = arg_.datav
    if arg_.val_data is not None:
        args['val_data'] = arg_.val_data
    else:
        args['val_data'] = args['dataset_version']
    args['find_lr'] = arg_.find_lr
    if arg_.lr:
        args['learning_rate'] = arg_.lr

    if hparam.dataset == 'wmt':
        # if args['dataset_version'] == 'full':
        #     val_check_interval = 2500
        # else:
        #     val_check_interval = 500
        val_check_interval = 2500
    else:
        val_check_interval = 1.0
    args['val_check_interval'] = val_check_interval

    #Logging into WANDB if needed
    if hparam.wandb_log:
        wandb.init(project=hparam.wandb_project, name=f"{hparam.method}_{args['dataset_version']}" , config=args, settings=wandb.Settings(start_method="fork"))
        wandb_logger = WandbLogger(project=hparam.wandb_project)
        wandb.define_metric("em_score", summary="max")
        wandb.define_metric("f1_score", summary="max")

        args = wandb.config.as_dict()
    else:
        wandb_logger = None 

    args = argparse.Namespace(**args)

    if args.output_dir=="":
        checkpoint_callback = False # Do not save model checkpoints when output dir is empty
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
            if args.dataset_version == 'full':
                # _________________________________Debug_____________
                every_n_train_steps=10
                # every_n_train_steps=2500
            else:
                # _________________________________Debug_____________
                every_n_train_steps=10
            callbacks = [ModelCheckpoint(dirpath = args.output_dir, filename = '{epoch}-{f1_score:.4f}-{em_score:.4f}', save_top_k=2, every_n_train_steps=every_n_train_steps, mode="max", monitor="f1_score")]
        elif args.dataset == 'nyt':
            if args.dataset_version == 'full':
                every_n_train_steps=2500
                callbacks = [ModelCheckpoint(dirpath = args.output_dir, filename = '{epoch}-{f1_score:.4f}-{em_score:.4f}', save_top_k=2, every_n_train_steps=every_n_train_steps, mode="max", monitor="f1_score")]
            else:
                callbacks = [ModelCheckpoint(dirpath = args.output_dir, filename = '{epoch}-{f1_score:.4f}-{em_score:.4f}', save_top_k=2, period=1, mode="max", monitor="f1_score")]
        elif args.dataset_version == 'debug':
            every_n_train_steps=1
            callbacks = [ModelCheckpoint(dirpath = args.output_dir, filename = '{epoch}-{f1_score:.4f}-{em_score:.4f}', save_top_k=2, every_n_train_steps=every_n_train_steps, mode="max", monitor="f1_score")]
        else:
            callbacks = [ModelCheckpoint(dirpath = args.output_dir, filename = '{epoch}-{f1_score:.4f}-{em_score:.4f}', save_top_k=2, period=1, mode="max", monitor="em_score")]
    checkpoint_callback = True

    if args.checkpoint_dir is not None:
        # args.checkpoint_dir += args.method
        # args.checkpoint_dir += 'baseline'
        # args.checkpoint_dir += '_' + str(args.dataset_version)
        # args.checkpoint_dir += '_' + '2freeze'
        # args.checkpoint_dir += '_' + ''.join(map(str, args.adapter_list))
        # args.checkpoint_dir += '_' + str(args.adapter_hidden_size)
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

    if arg_.checkpoint_path is not None:
        args.checkpoint_path = arg_.checkpoint_path

    if args.method == 'kadapter_ensemble':
        args.years_to_paths = {}
        for year in range(args.year_start, args.year_end+1):
            dir = f"outputs/wmtkadapter_{year}_2freeze_{''.join(map(str, args.adapter_list))}_{args.adapter_hidden_size}"
            args.years_to_paths[str(year)] = extract_adapters(dir)

    args.adapter_config = {'adapter_list': args.adapter_list, 'adapter_hidden_size': args.adapter_hidden_size, 'adapter_enc_dec': args.adapter_enc_dec, 'pool_size': args.pool_size, 'years_to_paths': args.years_to_paths, 'load_adapters': args.load_adapters}
    
    print(args)

    # Logging Learning Rate Scheduling
    if args.use_lr_scheduling and hparam.wandb_log:
        callbacks.append(pl.callbacks.LearningRateMonitor())

    if args.use_deepspeed:
        plugins = 'deepspeed_stage_2'
        use_fp_16 = True
    else:
        plugins = []
        use_fp_16 = False

    # Setting Flags for pytorch lightning trainer. Details: https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html#trainer-flags
    train_params = dict(
        plugins=plugins,
        gpus=args.n_gpu,
        max_epochs=args.num_train_epochs,
        precision= 16 if use_fp_16 else 32,
        amp_level=args.opt_level,
        resume_from_checkpoint=args.checkpoint_path if args.resume_from_checkpoint else None,
        gradient_clip_val=args.max_grad_norm,
        checkpoint_callback=checkpoint_callback,
        val_check_interval=args.val_check_interval,
        logger=wandb_logger,
        callbacks = callbacks,
        accelerator="ddp",
        auto_lr_find=args.find_lr
    )

    Model = load_model(type='T5')

    if args.check_validation_only:
    #    evaluate(args, hparam.wandb_project, hparam.wandb_run_name, Model)
        # set_seed(40)
        if args.checkpoint_path!="":
            model = load_checkpoint(Model, args)
        else:
            model = Model(args)
        trainer = pl.Trainer(**train_params)
        trainer.validate(model)
    else:
        # set_seed(40)
        if args.checkpoint_path!="":
            model = load_checkpoint(Model, args)
        else:
            model = Model(args)
        trainer = pl.Trainer(**train_params)
        if args.find_lr:
            trainer.tune(model)
            wandb.config.update({'learning_rate': model.hparams.learning_rate}, allow_val_change=True)
            wandb.update()
        trainer.fit(model)
        
def load_checkpoint(Model, args):
    model = Model.load_from_checkpoint(checkpoint_path=args.checkpoint_path, hparams=args, strict=False)
    if args.resume_from_checkpoint: 
        checkpoint = torch.load(args.checkpoint_path)
        model.dataloader = copy.deepcopy(checkpoint['dataloader'])
    return model

if __name__ == '__main__':
    main()