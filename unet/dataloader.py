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
    

if __name__ == '__main__':
    EPOCHS = 32
    BATCH_SIZE = 4
    PATH = '/home/deveshdatwani/Datasets/Caravan'
    HEIGHT=500
    WIDTH=500
    dataset = Caravan(PATH)
    dataloader = Loader()
    data_loader = dataloader(dataset=dataset, batch_size=BATCH_SIZE)
    sample = next(iter(data_loader))
    images, masks = sample.values()
    
    f = plt.figure()
    N = 4

    for i in range(N):
        f.add_subplot(1, N, i+1)
        plt.imshow(np.asarray(images[i].permute(1,2,0), dtype=np.uint8))

    for i in range(4):
        f.add_subplot(2, N, i+1)
        plt.imshow(np.asarray(masks[i], dtype=np.uint8))

    plt.show()
    