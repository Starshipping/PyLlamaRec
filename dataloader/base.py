from abc import *
import random


class AbstractDataloader(metaclass=ABCMeta):
    def __init__(self, args, dataset):
        self.args = args
        self.save_folder = dataset._get_preprocessed_folder_path()
        dataset = dataset.load_dataset()
        sel