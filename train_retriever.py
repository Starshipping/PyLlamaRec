import os
import torch
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import wandb
import argparse

from config import *
from model import *
from dataloader import *
from trainer import *

from pytorch_lightning import seed_everything

try:
    os.environ['WANDB_PROJECT'] = PROJECT_NAME
except:
    print('WANDB_PROJECT not available, please set it in config.py')


def main(args, export_root=None):
    seed_everything(args.seed)
    train_loader, val_loader, test_loader = dataloader_factory(args)
    model = LRURec(args)
    if export_root == None:
        export_root = EXPERIMENT_ROOT + '/' + args.model_code + '/' + args.dataset_code
    
    trainer = LRUTrainer(args, model, train_loader, val_loader, test_loader, export_root, args.use_wandb)
    trainer.train()
    trainer.test()

    # the next line generates val / test candidates for reranking
    trainer.generate_candidates(os.path.join(export_root, 'retrieved.pkl'))


if