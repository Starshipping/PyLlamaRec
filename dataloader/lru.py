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
        dataset = dataset.load_dataset()
        self.train = dataset['train']
        self.val = dataset['val']
        self.test = dataset['test']
        self.umap = dataset['umap']
        self.smap = dataset['smap']
        self.user_count = len(self.umap)
        self.item_count = len(self.smap)

        args.num_users = self.user_count
        args.num_items = self.item_count
        self.max_len = args.bert_max_len
        self.sliding_size = args.sliding_window_size

    @classmethod
    def code(cls):
        return 'lru'

    def get_pytorch_dataloaders(self):
        train_loader = self._get_train_loader()
        val_loader = self._get_val_loader()
        test_loader = self._get_test_loader()
        return train_loader, val_loader, test_loader
    
    def get_pytorch_test_subset_dataloader(self):
        retrieved_file_path = self.args.llm_retrieved_path
        print('Loading retrieved file from {}'.format(retrieved_file_path))
        retrieved_file = pickle.load(open(os.path.join(retrieved_file_path,
                                                       'retrieved.pkl'), 'rb'))
        
        test_probs = retrieved_file['test_probs']
        test_labels = retrieved_file['test_labels']
        test_users = [u for u, (p, l) in enumerate(zip(test_probs, test_labels), start=1) \
                      if l in torch.topk(torch.tensor(p), self.args.llm_negative_sample_size+1).indices]

        dataset = dataset = LRUTestDataset(self.args, self.train, self.val, self.test, self.max_len, 
                                           self.rng, subset_users=test_users)
        dataloader = data_utils.DataLoader(dataset, batch_size=self.args.val_batch_size, shuffle=False,
                                           pin_memory=True, num_workers=self.args.num_workers)
        return dataloader

    def _get_train_loader(self):
        dataset = self._get_train_dataset()
        dataloader = data_utils.DataLoader(dataset, batch_size=self.args.train_batch_size,
            