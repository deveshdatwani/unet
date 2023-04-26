import os
import torch
import numpy as np
from PIL import Image
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, Resize


class ResizeSample(object):
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.resizer = Resize(width, height)
    
    def __call__(self, sample):
        return {'image': self.resizer(sample['image']), 'mask': self.resizer(sample['mask'])} 


class ToTensor(object):
    def __init__(self):
        pass

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        image = image.transpose((2,0,1))

        return {'image': image, 'mask':mask}


class Caravan(Dataset):
    def __init__(self, root_dir, json=None):
       self.root_dir = root_dir
       self.train_images_dir = os.path.join(root_dir, "train")
       self.train_images = os.listdir(self.train_images_dir)
       self.sample_transform = transforms.Compose([ToTensor()])


    def __len__(self):
        return len(os.listdir(self.train_images_dir))


    def transform(self, sample):
        return {'image': self.sample_transform(sample['image']), 'mask': self.sample_transform(sample['mask'])}
    

    def get_mask_address(self, image_address):
        image_name = image_address.split('/')[-1]
        mask_address = self.root_dir + '/train_masks/' + image_name[:-4] + '_mask.gif'
        return mask_address


    def __getitem__(self, idx):
        image_address = os.path.join(self.train_images_dir, self.train_images[idx])
        mask_address = self.get_mask_address(image_address)       

        image = np.asarray(Image.open(image_address), dtype=np.uint8)
        mask = np.asanyarray(Image.open(mask_address), dtype=np.uint8)
        sample = {'image': image, 'mask': mask}

        sample = self.sample_transform(sample)
        
        return sample
