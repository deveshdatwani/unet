import cv2
import torch 
import numpy as np 
from time import time
from PIL import Image
from matplotlib import pyplot as plt
from torchvision.transforms import ToTensor, Resize, InterpolationMode, Normalize


def infer(image_address, model):
    image_size = [572, 572]
    image = np.array(cv2.imread(image_address), dtype=np.float32) / 255.0
    plt.imshow(image)
    plt.show()

    image = image.transpose((2,0,1)) 
    image = torch.from_numpy(image)
    image = Resize(image_size, interpolation=InterpolationMode.BILINEAR, antialias=True )(image)
    start = time()
    inference = model(image.unsqueeze(0))
    end = time()
    print(f'total time {end-start}')
    plt.imshow(inference[0].detach().numpy().transpose(1,2,0))
    plt.show()

    return None 