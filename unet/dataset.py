import os
import torch
import numpy as np
from PIL import Image
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import Dataset
from matplotlib import pyplot as plt
from torchvision.transforms import ToTensor, Resize, InterpolationMode, Normalize


class ResizeSample(object):
    def __init__(self):
        self.image_size = [572, 572]
        self.mask_size = [388, 388]

    def __call__(self, sample):
        image = sample['image']
        mask = sample['mask']
        
        image = Resize(self.image_size, interpolation=InterpolationMode.BILINEAR, antialias=True )(image)
        mask = Resize(self.mask_size, interpolation=InterpolationMode.BILINEAR, antialias=True )(mask) 
        
        sample = {'image': image, 'mask': mask}
        
        return sample 


class ToTensor(object):
    def __init__(self):
        pass

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        image = image.transpose((2,0,1))    

        return {'image': torch.from_numpy(image), 'mask':torch.from_numpy(mask).unsqueeze(0)}


class NormalizeImage():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    
    def __call__(self, sample):
        image = sample['image'] 
        mask = sample['mask']

        image_normalized = Normalize(self.mean, self.std)(image)
        mask_normalized = mask 
        # Normalize(self.mean, self.std)(mask)
        sample = {'image': image_normalized, 'mask': mask_normalized}

        return sample

class Caravan(Dataset):
    def __init__(self, root_dir, json=None):
       self.root_dir = root_dir
       self.train_images_dir = os.path.join(root_dir, "train")
       self.train_images = os.listdir(self.train_images_dir)
       self.sample_transform = transforms.Compose([ToTensor(), ResizeSample()])


    def __len__(self):
        return len(os.listdir(self.train_images_dir))


    def transform(self, sample):
        sample = self.sample_transform(sample)

        return sample 
    

    def get_mask_address(self, image_address):
        image_name = image_address.split('/')[-1]
        mask_address = self.root_dir + '/train_masks/' + image_name[:-4] + '_mask.gif'

        return mask_address


    def __getitem__(self, idx):
        image_address = os.path.join(self.train_images_dir, self.train_images[idx])
        mask_address = self.get_mask_address(image_address)   
        image = np.array(Image.open(image_address), dtype=np.float32) / 255.0
        mask = np.array(Image.open(mask_address), dtype=np.float32)
        
        sample = {'image': image, 'mask': mask}
        sample = self.sample_transform(sample)
        
        return sample
