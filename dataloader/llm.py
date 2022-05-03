from .base import AbstractDataloader
from .utils import Prompter

import torch
import random
import numpy as np
import torch.utils.data as data_utils

import os
import pickle
import transformers
from transformers import AutoTokenizer
from transformers.models.llama.tokenization_llama import DEFAULT_SYSTEM_PROMPT
from trainer import absolute_recall_mrr_ndcg_for_ks


def worker_init_fn(worker_id):
    r