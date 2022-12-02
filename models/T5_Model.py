import pytorch_lightning as pl
from models.Modular_T5 import T5ForConditionalGeneration as T5_Modular
from models.Modular_Small_T5 import T5ForConditionalGeneration as T5_Modular_Small
from models.Modular_Small_T52 import T5ForConditionalGeneration as T5_Modular_Small2
from models.Kadapter_T5 import T5ForConditionalGeneration as T5_Kadapter
from models.Kadapter_T52 import T5ForConditionalGeneration as T5_Kadapter2
from models.Kadapter_T5_parallel import T5ForConditionalGeneration as T5_Kadapter_parallel
from models.Kadapter_T5_soft import T5ForConditionalGeneration as T5_Kadapter_soft
from models.Kadapter_T5_ensemble import T5ForConditionalGeneration as T5_Kadapter_ensemble
from models.Padapter_T5 import T5ForConditionalGeneration as T5_Padapter
from models.Padapter2_T5 import T5ForConditionalGeneration as T5_Padapter2
from models.Lora_T5 import T5ForConditionalGeneration as T5_Lora
from models.Lora_T52 import T5ForConditionalGeneration as T5_Lora2
from models.RecAdam import RecAdam
import torch.nn.functional as F
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
)
import torch
from Datasets import Pretrain
from torch.utils.data import RandomSampler
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim import AdamW
from rouge import Rouge
from collections import Counter
import wandb
import models.my_sampler as my_sampler
import models.my_sampler_year_batching as my_sampler_year_batching

import re
import string
import copy

class T5(pl.LightningModule):
    def __init__(self, hparams):
        super(T5, self).__init__()
        self.save_hyperparameters(hparams)

        self.mix_ratio = 4
        self.mix_decay = 0.7

        if hparams.method=='modular':
            self.model = T5_Modular.from_pretrained(hparams.model_name_or_path)
        elif hparams.method=='modular_small':
            self.model = T5_Modular_Small.from_pretrained(hparams.model_name_or_path)
        elif hparams.method=='modular_small2': 
            previous_model_dir = (hparams.output_dir)[:len(hparams.output_dir)-1]
            self.model = T5_Modular_Small2.from_pretrained(previous_model_dir)
        elif hparams.method=='kadapter':
            self.model = T5_Kadapter.from_pretrained(hparams.model_name_or_path, hparams.adapter_config, cache_dir = 'huggingface')
        elif hparams.method=='kadapter2':
            previous_model_dir = (hparams.output_dir)[:len(hparams.output_dir)-1]
            self.model = T5_Kadapter2.from_pretrained(previous_model_dir)
        elif hparams.method=='kadapter_parallel':
            self.model = T5_Kadapter_parallel.from_pretrained(hparams.model_name_or_path)
        elif hparams.method=='kadapter_soft':
            self.model = T5_Kadapter_soft.from_pretrained(hparams.model_name_or_path, hparams.adapter_config, cache_dir = 'huggingface')
        elif hparams.method=='kadapter_ensemble':
            self.model = T5_Kadapter_ensemble.from_pretrained(hparams.model_name_or_path, hparams.adapter_config, cache_dir = 'huggingface')
        elif hparams.method=='padapter':
            self.model = T5_Padapter.from_pretrained(hparams.model_name_or_path, hparams.adapter_config, cache_dir = 'huggingface')
        elif hparams.method=='padapter2':
            self.model = T5_Padapter2.from_pretrained(hparams.model_name_or_path, hparams.adapter_config, cache_dir = 'huggingface')
        elif hparams.method=='lora':
            self.model = T5_Lora.from_pretrained(hparams.model_name_or_path)
        elif hparams.method=='lora2':
            previous_model_dir = (hparams.output_dir)[:len(hparams.output_dir)-1]
            self.model = T5_Lora2.from_pretrained(previous_model_dir)
        elif hparams.method=='recadam':
            self.model = T5ForConditionalGeneration.from_pretrained(hparams.model_name_or_path)
            self.pretrained_model = T5ForConditionalGeneration.from_pretrained(hparams.model_name_or_path)
            self.freeze_params(self.pretrained_model) #Freezing pretrained model
        elif hparams.method=='recadam2':
            self.model = T5ForConditionalGeneration.from_pretrained(hparams.model_name_or_path)
            self.pretrained_model = T5ForConditionalGeneration.from_pretrained(hparams.model_name_or_path)
            self.freeze_params(self.pretrained_model) #Freezing pretrained model
        else:
            self.model = T5ForConditionalGeneration.from_pretrained(hparams.model_name_or_path, cache_dir = 'huggingface')
        self.tokenizer = T5Tokenizer.from_pretrained(hparams.tokenizer_name_or_path)
        
        #Freezing only encoder or the whole model
        if hparams.freeze_level==0: # Do not freeze any parameters
            print('Not freezing any parameters!')
        elif hparams.freeze_level==1: # Freeze encoder only
            self.freeze_params(self.model.get_encoder())
        elif hparams.freeze_level==2: # Freeze encoder and decoder
            self.freeze_params(self.model) 

        if hparams.method=='modular_small':
            for name, param in self.model.named_parameters():
                if 'encoder_modular' in name:
                    param.requires_grad = True
        elif hparams.method=='modular_small2':
            for name, param in self.model.named_parameters():
                if 'encoder_modular2' in name or name=='encoder_modular_projection':
                    param.requires_grad = True
        elif hparams.method=='kadapter':
            # Unfreezing the parameters used for lora
            for name, param in self.model.named_parameters():
                if 'kadapter' in name:
                    param.requires_grad = True
                    # print(name)
            # print(self.state_dict().keys())
        elif hparams.method=='kadapter_soft':
            # Unfreezing the parameters used for lora
            for name, param in self.model.named_parameters():
                if 'kadapter' in name:
                    param.requires_grad = True
        elif hparams.method=='kadapter_ensemble':
            # Unfreezing the parameters used for lora
            for name, param in self.model.named_parameters():
                if 'kadapter' in name:
                    param.requires_grad = True
        elif hparams.method=='lora':
            # Unfreezing the parameters used for lora
            for name, param in self.model.named_parameters():
                if 'lora' in name:
                    param.requires_grad = True
        elif hparams.method=='kadapter2':
            # Unfreezing the parameters used for lora
            for name, param in self.model.named_parameters():
                if 'kadapter2' in name:
                    param.requires_grad = True
        elif hparams.method=='padapter':
            # Unfreezing the parameters used for lora
            for name, param in self.model.named_parameters():
                if 'adapter' in name:
                    param.requires_grad = True
        elif hparams.method=='padapter2':
            # Unfreezing the parameters used for lora
            for name, param in self.model.named_parameters():
                if 'adapter' in name:
                    param.requires_grad = True
        elif hparams.method=='lora2':
            # Unfreezing the parameters used for lora
            for name, param in self.model.named_parameters():
                if 'lora2' in name:
                    param.requires_grad = True
 
        self.output_dir = self.hparams.output_dir
            
        n_observations_per_split = {
            "train": self.hparams.n_train,
            "validation": self.hparams.n_val,
            "test": self.hparams.n_test,
        }
        self.n_obs = {k: v if v >= 0 else None for k, v in n_observations_per_split.items()}

    def normalize_answer(self, s):
        """Lower text and remove punctuation, articles and extra whitespace."""

        def remove_articles(text):
            return re.sub(r"\b(a|an|the)\b", " ", text)

        def white_space_fix(text):
            return " ".join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return "".join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()
        
        def rid_of_specials(text):
            text = text.replace("<extra_id_0>", "")
            text = text.replace("<extra_id_1>", "")
            return text

        return rid_of_specials(white_space_fix(remove_articles(remove_punc(lower(s)))))

    def exact_match_score(self, prediction, ground_truth):
        return int(self.normalize_answer(prediction) == self.normalize_answer(ground_truth))
    
    def accuracy_match_score(self, prediction, ground_truth):
        return int(prediction.strip() == ground_truth.strip())

    def _rougel_score(self, prediction, ground_truth):
        rouge = Rouge()
        # no normalization
        try:
            scores = rouge.get_scores(prediction, ground_truth, avg=True)
        except ValueError:  # "Hypothesis is empty."
            return 0.0
        return scores["rouge-l"]["f"]
    
    def _f1_score(self, prediction, ground_truth):
        prediction_tokens = self.normalize_answer(prediction).split()
        ground_truth_tokens = self.normalize_answer(ground_truth).split()
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return 0
        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1

    def calculate_scores(self, predictions, ground_truths):
        em_score = 0
        accuracy = 0
        
        for i in range(len(predictions)):
            ground_truth = ground_truths[i]
            prediction = predictions[i]
            em_score +=  self.exact_match_score(prediction, ground_truth)
            accuracy += self.accuracy_match_score(prediction, ground_truth)
        
        em_score /= len(predictions)
        accuracy /= len(predictions)
        return em_score*100, accuracy*100
    
    def calculate_scores_multipleanswers(self, predictions, ground_truths, ids):
        em_score = 0
        accuracy_score = 0
        
        for i in range(len(predictions)):
            unique_id = ids[i]
            answers = self.ids_to_answers[unique_id]
            prediction = predictions[i]
            em_correct = False
            accuracy_correct = False
            for answer in answers:
                accuracy = self.accuracy_match_score(prediction, answer)
                if accuracy == 1:
                    accuracy_correct = True
                em  = self.exact_match_score(prediction, answer)
                if em == 1:
                    em_correct = True
            if accuracy_correct:
                accuracy_score+=1
            if em_correct:
                em_score+=1
        
        accuracy_score /= len(predictions)
        em_score /= len(predictions)
        return em_score*100, accuracy_score*100

    def calculate_rouge_multipleanswers(self, predictions, ground_truths, ids):
        rouge_score = 0 
        for i in range(len(predictions)):
            unique_id = ids[i]
            answers = self.ids_to_answers[unique_id]
            prediction = predictions[i]
            rouge_local_score = 0
            for answer in answers:
                rouge = self._rougel_score(prediction, answer)
                if rouge > rouge_local_score:
                    rouge_local_score = rouge 
            rouge_score += rouge_local_score
        rouge_score /= len(predictions)
        return rouge_score*100

    def calculate_f1_scores(self, predictions, ground_truths, ids):
        f1_score = 0 
        for i in range(len(predictions)):
            ground_truth = ground_truths[i]
            prediction = predictions[i]
            f1_score += self._f1_score(prediction, ground_truth)

        f1_score /= len(predictions)
        return f1_score*100

    def get_dataset(self, tokenizer, type_path, num_samples, args, length=None):
        if args.mode == 'pretrain' or args.mode == 'finetune':
            dataset = Pretrain(tokenizer=tokenizer, type_path=type_path, num_samples=num_samples,  input_length=args.max_input_length, 
                            output_length=args.max_output_length, args=args, length=length)
            if dataset.ids_to_answers is not None:
                self.ids_to_answers = dataset.ids_to_answers
            return dataset
        else:
            raise NameError('Select the correct mode please.')

    def freeze_params(self, model):
        for par in model.parameters():
            par.requires_grad = False
             
    def lmap(self, f, x):
        """list(map(f, x))"""
        return list(map(f, x))
    

    def is_logger(self):
        return self.trainer.global_rank <= 0
    
    def forward(self, input_ids, attention_mask=None, decoder_input_ids=None, decoder_attention_mask=None, lm_labels=None, year=None):
        if self.hparams.method in ['kadapter_soft', 'kadapter_ensemble']:
            return self.model(
                input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                labels=lm_labels,
                year=year
            )
        else:        
            return self.model(
                input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                labels=lm_labels,
            )

    def _step(self, batch):
        lm_labels = batch["target_ids"]
        lm_labels[lm_labels[:, :] == self.tokenizer.pad_token_id] = -100
        
        if self.hparams.method in ['kadapter_soft', 'kadapter_ensemble']:
            outputs = self(
                input_ids=batch["source_ids"],
                attention_mask=batch["source_mask"],
                lm_labels=lm_labels,
                decoder_attention_mask=batch['target_mask'],
                year = batch['year']
            )
        else:
            outputs = self(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            lm_labels=lm_labels,
            decoder_attention_mask=batch['target_mask'],
        )
        
        loss = outputs[0]
        return loss
    
    
    def ids_to_clean_text(self, generated_ids, skip_special_tokens):
        gen_text = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=skip_special_tokens, clean_up_tokenization_spaces=True
        )
        return self.lmap(str.strip, gen_text)
    
     
    def _generative_step(self, batch):     
        if self.hparams.method in ['kadapter_soft', 'kadapter_ensemble']:
            generated_ids = self.model.generate(
                batch["source_ids"],
                attention_mask=batch["source_mask"],
                use_cache=True,
                decoder_attention_mask=batch['target_mask'],
                max_length=self.hparams.max_output_length,
                num_beams=2,
                early_stopping=True,
                year = batch['year']
            )
        else:
            generated_ids = self.model.generate(
                batch["source_ids"],
                attention_mask=batch["source_mask"],
                use_cache=True,
                decoder_attention_mask=batch['target_mask'],
                max_length=self.hparams.max_output_length,
                num_beams=2,
                early_stopping=True
            )
        
        preds = self.ids_to_clean_text(generated_ids, skip_special_tokens=True)
        return preds

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)
        # DEBUG
        # print(f"----------batch_idx = {batch_idx}----------")
        self.log("loss", loss)
        return loss

    # def on_train_start(self) -> None:
    #     print("----------train start----------")

    def on_train_epoch_start(self):
        print(f"cuda memory allocated: {torch.cuda.max_memory_allocated()}")
        # print("----------epoch start----------")
        self.sampler.set_mid_epoch = True

    def on_train_epoch_end(self):
        # print("----------epoch end----------")
        self.sampler.idx = 0
        self.sampler.mid_epoch = False

    def on_train_batch_start(self, batch, batch_idx: int, dataloader_idx: int) -> None:
        # print(f"cuda memory allocated: {torch.cuda.max_memory_allocated()}")
        # print("----------batch start----------")
        self.sampler.idx += self.hparams.train_batch_size
        # print(f"self.sampler.idx = {self.sampler.idx}. after this batch completes, this is the next data index")

    # def on_train_batch_end(self, outputs, batch, batch_idx: int, dataloader_idx: int) -> None:
    #     print("----------batch end----------")

    def on_train_end(self):
        if self.hparams.mode == 'pretrain':
            if self.hparams.method=='recadam':
                self.pretrained_model = self.model
            # elif self.hparams.method=='kadapter' or self.hparams.method=='lora' or self.hparams.method=='modular_small':
            #     self.model.save_pretrained(self.hparams.output_dir)

    def on_validation_epoch_start(self) -> None:
        columns=["id", "question", "truth", "prediction", "f1"]
        if self.hparams.wandb_log:
            self.test_table = wandb.Table(columns=columns)
        self.table_len = 0

    def on_save_checkpoint(self, checkpoint) -> None:
        if self.sampler.set_mid_epoch:
            self.sampler.mid_epoch = True
            self.sampler.set_mid_epoch = False
        # Assumes last batch is dropped if truncated
        if self.sampler.idx + self.hparams.train_batch_size > len(self.sampler): 
            self.sampler.idx = 0
            self.sampler.mid_epoch = False
        if self.sampler.mid_epoch:
            checkpoint['epoch'] -= 1
        print(f"saving checkpoint, epoch ={checkpoint['epoch']}")
        checkpoint['sampler'] = self.sampler
    
    def validation_step(self, batch, batch_idx):
        total_cnt = len(batch['source_ids'])
        em_correct_num = 0
        f1_total = 0
        preds = self._generative_step(batch)
        texts = self.ids_to_clean_text(batch["source_ids"], skip_special_tokens=True)
        for i in range(total_cnt):
            predicted = preds[i]
            ids = batch['label_ids'][i].item()
            lines = texts[i]

            answer_list = self.ids_to_answers[str(ids)]
            em_correct = False
            f1 = 0
            for answer in answer_list:
                em = self.exact_match_score(predicted, answer)
                if em == 1:
                    em_correct = True
                f1 = max(f1, self._f1_score(predicted, answer))
            if em_correct:
                em_correct_num+=1
                if f1 != 1:
                    print(f"Error: f1 should be 1, is {f1}")
                    print(lines, answer_list, predicted)
            f1_total += f1
            # print(predicted, answer_list)
            # print(f"em = {em_correct}, f1 = {f1}")
            if self.hparams.wandb_log and self.table_len < 10000:
                self.test_table.add_data(ids, lines, answer_list, predicted, f1) 
                self.table_len += 1   
        f1 = f1_total / total_cnt
        em = em_correct_num / total_cnt
        self.log('em_score', em, prog_bar=True, on_epoch=True, logger=True)
        self.log('f1_score', f1, prog_bar=True, on_epoch=True, logger=True)

    def on_validation_epoch_end(self) -> None:
        if self.hparams.wandb_log:
            wandb.log({"table_key": self.test_table})

    def configure_optimizers(self, train_len=None):
        "Prepare optimizer and schedule (linear warmup and decay)"
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        if "adapter" in self.hparams.method and self.hparams.freeze_level < 2:
            adapter_params = [(n, p) for n, p in model.named_parameters() if "adapter" in n]
            adapters_params_names = set([n for n,p in adapter_params])
            optimizer_grouped_parameters = [
                {
                    "params": [p for n,p in adapter_params],
                    "lr": self.hparams.learning_rate
                },
                {
                    "params": [p for n, p in model.named_parameters() if n not in adapters_params_names],
                    "lr": self.hparams.t5_learning_rate
                },
            ]
            max_lr = [self.hparams.learning_rate, self.hparams.t5_learning_rate]
        else:
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": self.hparams.weight_decay,
                },
                {
                    "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                },
            ]
            max_lr = self.hparams.learning_rate
        self.optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate)

        #self.optimizer = optimizer
        if self.hparams.use_lr_scheduling:
            if self.hparams.dataset == 'wmt':
                len_data = len(self.train_dataloader())
                self.lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=max_lr, steps_per_epoch=len_data, epochs=self.hparams.num_train_epochs, pct_start=0.1)

                return [self.optimizer], [{"scheduler": self.lr_scheduler, "interval": "step", "name": "learning rate"}]
            elif self.hparams.dataset == 'templama':
                # self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[3])
                self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode = 'max', factor = 0.5, patience = 2)
                return [self.optimizer], [{"scheduler": self.lr_scheduler, "interval": "epoch", "name": "learning rate", "monitor": "em_score"}]
            else:
                len_data = len(self.train_dataloader())
                self.lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=max_lr, steps_per_epoch=len_data, epochs=self.hparams.num_train_epochs)
                    
                return [self.optimizer], [{"scheduler": self.lr_scheduler, "interval": "step", "name": "learning rate"}]
        else:
            return [self.optimizer]

    def train_dataloader(self):
        n_samples = self.n_obs['train']
        train_dataset = self.get_dataset(tokenizer=self.tokenizer, type_path="train", num_samples=n_samples, args=self.hparams)
        if not hasattr(self, 'sampler'):
            if self.hparams.method == 'kadapter_ensemble':
                self.sampler = my_sampler_year_batching.ResumableSampler(train_dataset, self.hparams.train_batch_size)
            else:
                self.sampler = my_sampler.ResumableSampler(train_dataset)
        self.dataloader = DataLoader(train_dataset, sampler=self.sampler,  batch_size=self.hparams.train_batch_size, drop_last=True, num_workers=self.hparams.num_workers)
        return self.dataloader

    def val_dataloader(self):
        n_samples = self.n_obs['validation']
        validation_dataset = self.get_dataset(tokenizer=self.tokenizer, type_path="validation", num_samples=n_samples, args=self.hparams)
        if self.hparams.method == 'kadapter_ensemble':
            sampler = my_sampler_year_batching.ResumableSampler(validation_dataset, self.hparams.eval_batch_size)
            return DataLoader(validation_dataset, sampler = sampler, batch_size=self.hparams.eval_batch_size, num_workers=self.hparams.num_workers)
        else:
            return DataLoader(validation_dataset, batch_size=self.hparams.eval_batch_size, num_workers=self.hparams.num_workers)
    
    def test_dataloader(self):
        n_samples = self.n_obs['test']
        test_dataset = self.get_dataset(tokenizer=self.tokenizer, type_path="test", num_samples=n_samples, args=self.hparams)
        
        return DataLoader(test_dataset, batch_size=self.hparams.eval_batch_size, num_workers=self.hparams.num_workers, shuffle=False)
        