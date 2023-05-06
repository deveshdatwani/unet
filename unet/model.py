import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from torchvision.transforms import ToTensor, Resize, InterpolationMode, Normalize

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, padding=1, kernel_size=3),
                                                   nn.BatchNorm2d(out_channels),
                                                   nn.ReLU(),
                                                   nn.Conv2d(out_channels, out_channels, padding=1, kernel_size=3),
                                                   nn.BatchNorm2d(out_channels),
                                                   nn.ReLU())


    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super(UNet, self).__init__()
        self.double_convolution = DoubleConv(3, 64)
        self.bottle_neck = None


    def forward(self, x):
        x = self.double_convolution(x)
        return x


if __name__ == '__main__':
    model  = UNet()
    input = torch.rand(size=(3, 572, 572))
    print(model(input.unsqueeze(0)).shape) 
            
