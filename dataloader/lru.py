from .base import AbstractDataloader

import os
import torch
import random
import pickle
import numpy as np
import torch.utils.data as data_utils


def worker_init_fn(worker_id):
    random.seed(np.random.get_state()[1][0] + worker_id)                                                      
    np.random.seed(np.random.get_state()[1][0] + worker_id)


class LRUDataloader():
    def __init__(self, args, dataset):
        self.args = args
        self.rng = np.random
        self.save_folder = dataset._get_preprocessed_folder_path()
        dataset = dataset.load_dat