from datasets import dataset_factory

from .lru import *
from .llm import *
from .utils import *


def dataloader_factory(args):
    dataset = dataset_factory(args)
    if args.model_code == 'lru':
        datal