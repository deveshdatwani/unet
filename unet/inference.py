import os
import torch
import argparse
from model import UNet
from utils import infer


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str)
    parser.add_argument('--image', type=int)
    args = parser.parse_args()


    torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    EPOCHS = 1
    BATCH_SIZE = 1
    DATA_PATH = '/home/deveshdatwani/Datasets/SMALL'
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
            checkpoint = torch.load(MODEL_WEIGHTS_ADDRESS, map_location=torch.device('cpu'))
            model.load_state_dict(checkpoint['model_state_dict'])

        else:
            print('NO CHECKPOINTS FOUND')

    images_address = os.path.join(DATA_PATH, os.listdir(DATA_PATH)[args.image])
    infer(images_address, model)