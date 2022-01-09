
import numpy as np
import random
import torch
import argparse


RAW_DATASET_ROOT_FOLDER = 'data'
EXPERIMENT_ROOT = 'experiments'
STATE_DICT_KEY = 'model_state_dict'
OPTIMIZER_STATE_DICT_KEY = 'optimizer_state_dict'
PROJECT_NAME = 'llmrec'


def set_template(args):
    if args.dataset_code == None:
        print('******************** Dataset Selection ********************')
        dataset_code = {'1': 'ml-100k', 'b': 'beauty', 'g': 'games'}
        args.dataset_code = dataset_code[input('Input 1 for ml-100k, b for beauty and g for games: ')]

    if args.dataset_code == 'ml-100k':
        args.bert_max_len = 200
    else:
        args.bert_max_len = 50

    if 'llm' in args.model_code: 
        batch = 16 if args.dataset_code == 'ml-100k' else 12
        args.lora_micro_batch_size = batch
    else: 
        batch = 16 if args.dataset_code == 'ml-100k' else 64

    args.train_batch_size = batch
    args.val_batch_size = batch
    args.test_batch_size = batch

    if torch.cuda.is_available(): args.device = 'cuda'
    else: args.device = 'cpu'
    args.optimizer = 'AdamW'
    args.lr = 0.001
    args.weight_decay = 0.01
    args.enable_lr_schedule = False
    args.decay_step = 10000
    args.gamma = 1.
    args.enable_lr_warmup = False
    args.warmup_steps = 100

    args.metric_ks = [1, 5, 10, 20, 50]
    args.rerank_metric_ks = [1, 5, 10]
    args.best_metric = 'Recall@10'
    args.rerank_best_metric = 'NDCG@10'

    args.bert_num_blocks = 2
    args.bert_num_heads = 2
    args.bert_head_size = None


parser = argparse.ArgumentParser()

################
# Dataset
################
parser.add_argument('--dataset_code', type=str, default=None)
parser.add_argument('--min_rating', type=int, default=0)
parser.add_argument('--min_uc', type=int, default=5)
parser.add_argument('--min_sc', type=int, default=5)
parser.add_argument('--seed', type=int, default=42)

################
# Dataloader
################
parser.add_argument('--train_batch_size', type=int, default=64)
parser.add_argument('--val_batch_size', type=int, default=64)
parser.add_argument('--test_batch_size', type=int, default=64)
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--sliding_window_size', type=float, default=1.0)
parser.add_argument('--negative_sample_size', type=int, default=10)

################
# Trainer
################
# optimization #