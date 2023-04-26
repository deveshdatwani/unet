from torch.utils.data import DataLoader
import torch
from torch import nn
from dataset import Caravan
from matplotlib import pyplot as plt
import numpy as np


class Loader(object):
    def __init__(self):
        pass

    def __call__(self, dataset, batch_size):
        return DataLoader(dataset, batch_size, shuffle=True)

    