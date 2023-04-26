from torch.utils.data import DataLoader
import torch
from torch import nn
from loss import DiceLoss
from torch.optim import SGD, Adam
from model import UNet
from tqdm import tqdm
from matplotlib import pyplot as plt
from dataset import Caravan
from dataloader import Loader


# TRAINING LOOP
class Trainer(object):
    def __init__(self, model=None, epochs=16, batch_size=2, path=None):
        self.model = model
        self.epochs = epochs
        self.batch_size = batch_size
        self.dataset = Caravan(PATH)
        self.dataloader = Loader()
        self.data_loader = self.dataloader(dataset=self.dataset, batch_size=self.batch_size)
        self.optim = Adam(params=model.parameters(), lr=0.0001)
        self.criterian = DiceLoss()


    def __call__(self):
        for epoch in tqdm(range(self.epochs)):
            running_loss = 0
            
            for i, data in enumerate(self.data_loader):
                instance, mask = data['image'], data['mask']
                prediction = model(instance.float())
                loss = self.criterian(prediction, mask)               
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                running_loss += loss.item()
                
                if i % 5 == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                         epoch, i * len(data), len(self.dataset),
                         100. * i / len(self.data_loader), running_loss / 5))
                    running_loss = 0
            
            break


if __name__ == "__main__":
    EPOCHS = 32
    BATCH_SIZE = 1
    PATH = '/home/deveshdatwani/Datasets/Caravan'
    model = UNet()
    trainer = Trainer(model=model, batch_size=BATCH_SIZE, epochs=EPOCHS, path=PATH)
    trainer()
    