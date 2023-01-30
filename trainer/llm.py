
from config import STATE_DICT_KEY, OPTIMIZER_STATE_DICT_KEY
from .verb import ManualVerbalizer
from .utils import *
from .loggers import *
from .base import *

import re
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import json
import numpy as np
from abc import *
from pathlib import Path

import bitsandbytes as bnb
from transformers.trainer import *
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback


def llama_collate_fn_w_truncation(llm_max_length, eval=False):
    def llama_collate_fn(batch):
        all_input_ids = []
        all_attention_mask = []
        all_labels = []
        example_max_length = max([len(batch[idx]['input_ids']) for idx in range(len(batch))])
        max_length = min(llm_max_length, example_max_length)
        
        for i in range(len(batch)):
            input_ids = batch[i]['input_ids']
            attention_mask = batch[i]['attention_mask']
            labels = batch[i]['labels']
            if len(input_ids) > max_length:
                input_ids = input_ids[-max_length:]
                attention_mask = attention_mask[-max_length:]
                if not eval: labels = labels[-max_length:]
            elif len(input_ids) < max_length:
                padding_length = max_length - len(input_ids)
                input_ids = [0] * padding_length + input_ids
                attention_mask = [0] * padding_length + attention_mask
                if not eval: labels = [-100] * padding_length + labels

            if eval: assert input_ids[-1] == 13
            else:
                assert input_ids[-3] == 13 and input_ids[-1] == 2
                assert labels[-3] == -100 and labels[-2] != -100
            
            all_input_ids.append(torch.tensor(input_ids).long())
            all_attention_mask.append(torch.tensor(attention_mask).long())
            all_labels.append(torch.tensor(labels).long())
        
        return {
            'input_ids': torch.vstack(all_input_ids),
            'attention_mask': torch.vstack(all_attention_mask),
            'labels': torch.vstack(all_labels)
        }
    return llama_collate_fn


def compute_metrics_for_ks(ks, verbalizer):
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        logits = torch.tensor(logits)
        labels = torch.tensor(labels).view(-1)
        scores = verbalizer.process_logits(logits)
        metrics = absolute_recall_mrr_ndcg_for_ks(scores, labels, ks)
        return metrics
    return compute_metrics


class LLMTrainer(Trainer):
    def __init__(
            self,
            args,
            model,
            train_loader,
            val_loader,
            test_loader,
            tokenizer,
            export_root,
            use_wandb,
            **kwargs
        ):
        self.original_args = args
        self.export_root = export_root
        self.use_wandb = use_wandb
        self.llm_max_text_len = args.llm_max_text_len
        self.rerank_metric_ks = args.rerank_metric_ks
        self.verbalizer = ManualVerbalizer(