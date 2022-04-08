from abc import *
import random


class AbstractDataloader(metaclass=ABCMeta):
    def __init__(self, args,