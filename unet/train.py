import os
import torch
from torch import nn
from tqdm import tqdm
from model import UNet
from loss import DiceLoss
from dataset import Caravan
from dataloader import Loader
from torch.optim import SGD, Adam
from infer import display_inference
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader


# TRAINING LOOP
class Trainer(object):
    def __init__(self, model=None, epochs=16, batch_size=2, path=None, checkpoint_path=''):
        self.model = model
        self.epochs = epochs
        self.batch_size = batch_size
        self.dataset = Caravan(PATH)
        self.dataloader = Loader()
        self.data_loader = self.dataloader(dataset=self.dataset, batch_size=self.batch_size)
        self.optim = SGD(params=self.model.parameters(), lr=0.001, momentum=0.99)
        self.criterian = DiceLoss()
        self.checkpoint_loss = 0
        # self.checkpoint_path = checkpoint_path
        self.loss_plot = []


    def __call__(self):
        DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f'DEVICE IS {DEVICE}')
        
        for epoch_number in tqdm(range(self.epochs)):
            running_loss = 0
            
            for i, data in enumerate(self.data_loader):
                instance, mask = data['image'], data['mask']
                instance = instance.to(device=DEVICE)
                mask = mask.to(device=DEVICE)
                
                # self.optim.zero_grad()
                prediction = self.model(instance)
                
                # loss = self.criterian(prediction, mask)               
                # loss.backward()
                # self.optim.step()
                
                # running_loss += loss.item()
                # self.checkpoint_loss += loss.item()
                # self.loss_plot.append(loss.item())
              
                display_inference(prediction, mask, instance)
                
                if i % 5 == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                         epoch_number + 1, i * len(data), len(self.dataset),
                         100. * i / len(self.data_loader), running_loss / 5))
                    running_loss = 0

                # if i % 20 == 0:
                #     torch.save({'epoch': epoch_number,
                #                 'model_state_dict': model.state_dict(),
                #                 'optimizer_state_dict': self.optim.state_dict(),
                #                 'loss': self.checkpoint_loss / 20,
                #                 }, self.checkpoint_path)
                #     self.checkpoint_loss = 0


if __name__ == "__main__":
    torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    EPOCHS = 1
    BATCH_SIZE = 1
    PATH = '/home/deveshdatwani/Datasets/Caravan'
    MODEL_TYPE = 'attention'
    model = UNet()
    MODEL_WEIGHTS_DIR = '/home/deveshdatwani/Unet/weights'

    if MODEL_TYPE == 'attention':
        MODEL_WEIGHTS_ADDRESS = '/home/deveshdatwani/Unet/weights/attention-model.pt'
        if 'attention-model.pt' in os.listdir(MODEL_WEIGHTS_DIR):
            checkpoint = torch.load(MODEL_WEIGHTS_ADDRESS, map_location=torch.device('cpu'))
            model.load_state_dict(checkpoint['model_state_dict'])
        
        else:
            print('NO CHECKPOINTS FOUND')
        
    else:
        MODEL_WEIGHTS_ADDRESS = '/home/deveshdatwani/Unet/weights/model.pt'
        if 'model.pt' in os.listdir(MODEL_WEIGHTS_DIR):
            checkpoint = torch.load(MODEL_WEIGHTS_ADDRESS)
            model.load_state_dict(checkpoint['model_state_dict'])

        else:
            print('NO CHECKPOINTS FOUND')
    
    trainer = Trainer(model=model, batch_size=BATCH_SIZE, epochs=EPOCHS, path=PATH)
    trainer()
    
    