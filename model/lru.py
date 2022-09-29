import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


class LRURec(nn.Module):
    def __init__(s