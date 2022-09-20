from .ml_100k import ML100KDataset
from .beauty import BeautyDataset
from .games import GamesDataset

DATASETS = {
    ML100KDataset.code(): ML100KDataset,
    BeautyDataset.code(): BeautyDataset,
    