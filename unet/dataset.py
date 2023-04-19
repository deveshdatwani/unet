import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor, Resize
import matplotlib.pyplot as plt
import pandas as pd
import os
from matplotlib import pyplot as plt
import cv2
from torchvision.io import read_image, ImageReadMode as iomode


class Kvasir(Dataset):

    def __init__(self, root_dir, json=None):
       self.train_images_dir = os.path.join(root_dir, "images")
       self.masks_dir = os.path.join(root_dir, "masks")
       self.len = len(os.listdir(self.train_images_dir))
       self.train_images = [os.path.join(self.train_images_dir, i) for i in os.listdir(self.train_images_dir)]
       self.train_masks = [os.path.join(self.masks_dir, i) for i in os.listdir(self.masks_dir)]
       self.size = [572, 572]

    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        
        image_address = self.train_images[idx]
        mask_address = self.train_masks[idx]
        image = read_image(image_address)
        mask = read_image(mask_address, iomode.GRAY)
        image = Resize(self.size)(image)
        mask = Resize(self.size)(mask)

        return image, mask



if __name__ == '__main__':
    dataset = Kvasir('/home/deveshdatwani/Datasets/Kvasir-SEG')
    print("ALL OKAY") 
    image, mask = dataset[120]
    print(mask.shape)
    print(image.shape)