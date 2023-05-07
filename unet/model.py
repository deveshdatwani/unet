import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from torchvision.transforms import ToTensor, Resize, InterpolationMode, Normalize

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, padding=1, kernel_size=3),
                                    nn.BatchNorm2d(out_channels),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(out_channels, out_channels, padding=1, kernel_size=3),
                                    nn.BatchNorm2d(out_channels),
                                    nn.ReLU(inplace=True))


    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super(UNet, self).__init__()
        self.downsample = nn.MaxPool2d(2)
        self.double_convolution_1 = DoubleConv(3, 64)
        self.double_convolution_2 = DoubleConv(64, 128)
        self.double_convolution_3 = DoubleConv(128, 256)
        self.double_convolution_4 = DoubleConv(256, 512)
        self.double_convolution_5 = DoubleConv(512, 1024)
        self.upsample_5 = nn.ConvTranspose2d(1024, 512, 2, 2)
        self.up_double_convolution_5 = DoubleConv(1024, 512)
        self.bottle_neck = None
        

    def forward(self, x):
        x_skip_connection_1 = self.double_convolution_1(x)
        x = self.downsample(x_skip_connection_1)
        x_skip_connection_2 = self.double_convolution_2(x)
        x = self.downsample(x_skip_connection_2)
        x_skip_connection_3 = self.double_convolution_3(x)
        x = self.downsample(x_skip_connection_3)
        x_skip_connection_4 = self.double_convolution_4(x)
        x = self.downsample(x_skip_connection_4)
        x = self.double_convolution_5(x)
        x = self.upsample_5(x)
        
        if x.shape != x_skip_connection_4.shape:
            x = TF.resize(x, 71, antialias=None)
        
        x = torch.cat([x, x_skip_connection_4])
        print(x.shape)
        x = self.up_double_convolution_5(x)


        return x


if __name__ == '__main__':
    model  = UNet()
    input = torch.rand(size=(3, 572, 572))
    print(model(input.unsqueeze(0)).shape) 
            
