from torch.utils.data import DataLoader
import torch
from torch import nn
from dataset import Kvasir
from loss import DiceLoss
from torch.optim import SGD, Adam
from model import UNet
from tqdm import tqdm
from matplotlib import pyplot as plt


# TRAINING LOOP
class Trainer():
    def __init__(self, model=None, epochs=16, batch_size=2, path='/home/deveshdatwani/Datasets/Kvasir-SEG'):
        self.model = model
        self.epochs = epochs
        self.batch_size = batch_size
        self.dataset = Kvasir(path)
        self.train_dataloder = DataLoader(dataset=self.dataset, shuffle=True, batch_size=batch_size)
        self.optim = Adam(params=model.parameters(), lr=0.001)
        self.criterian = DiceLoss()


    def train(self):
        for i in tqdm(range(self.epochs)):
            running_loss = 0
            
            for i, data in enumerate(self.train_dataloder):
                instance, ground_truth = data
                prediction = model(instance.float())
                self.optim.zero_grad()
                loss = self.criterian(prediction, ground_truth)

                loss.backward()
                self.optim.step()
                running_loss += loss

                if i != 0 and i % 7 == 0:
                    print(f'Current losss {running_loss}')
                    running_loss = 0
                
            print(f'Dice Loss {running_loss}')
                



if __name__ == "__main__":
    model = UNet()
    trainer = Trainer(model=model)
    trainer.train()
    