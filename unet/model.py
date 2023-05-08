import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from torchvision.transforms import ToTensor, Resize, InterpolationMode, Normalize
from copy import copy


class AttentionBlock(nn.Module):
    def __init__(self, F):
        super(AttentionBlock, self).__init__()
        self.convWx = nn.Sequential(nn.Conv2d(F, F, kernel_size=1, bias=True), nn.BatchNorm2d(F))
        self.convWg = nn.Sequential(nn.Conv2d(F, F, kernel_size=1, bias=True), nn.BatchNorm2d(F))
        self.relu = nn.ReLU(inplace=True)
        self.convPsi = nn.Sequential(nn.Conv2d(F, 1, kernel_size=1, bias=True), nn.BatchNorm2d(1))
        self.sigmoid = nn.Sigmoid()


    def forward(self, Wg, Wx):
        Wx_add = copy(Wx)
        Wg = self.convWg(Wg)
        Wx = self.convWx(Wx)
        AG = Wg + Wx
        AG = self.relu(AG)
        AG = self.convPsi(AG)
        AG = self.sigmoid(AG)
        Wx = AG * Wx_add 


        return Wx 
    

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
        self.upsample_6 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.upsample_7 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.upsample_8 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.up_double_convolution_5 = DoubleConv(1024, 512)
        self.up_double_convolution_6 = DoubleConv(512, 256)
        self.up_double_convolution_7 = DoubleConv(256, 128)
        self.up_double_convolution_8 = DoubleConv(128, 64)
        self.final_conv = nn.Sequential(nn.Conv2d(64, 1, 1, 1), nn.Sigmoid())
        self.attention_block1 = AttentionBlock(512)
        self.attention_block2 = AttentionBlock(256)
        self.attention_block3 = AttentionBlock(128)
        

    def forward(self, x):
        # Downward 

        x_skip_connection_1 = self.double_convolution_1(x)
        x = self.downsample(x_skip_connection_1)
        x_skip_connection_2 = self.double_convolution_2(x)
        x = self.downsample(x_skip_connection_2)
        x_skip_connection_3 = self.double_convolution_3(x)
        x = self.downsample(x_skip_connection_3)
        x_skip_connection_4 = self.double_convolution_4(x)
        x = self.downsample(x_skip_connection_4)
        x = self.double_convolution_5(x)
        
        # Upward 
        x = self.upsample_5(x)
        
        if x.shape != x_skip_connection_4.shape:
            x_skip_connection_4 = TF.resize(x_skip_connection_4, x.shape[2], antialias=None)
        
        x = self.attention_block1(x_skip_connection_4, x)

        x = torch.cat([x, x_skip_connection_4], dim=1)
        x = self.up_double_convolution_5(x)
        x = self.upsample_6(x)

        if x.shape != x_skip_connection_3.shape:
            x_skip_connection_3 = TF.resize(x_skip_connection_3, x.shape[2], antialias=None)

        x = torch.cat([x, x_skip_connection_3], dim=1)
        x = self.up_double_convolution_6(x)
        x = self.upsample_7(x)

        if x.shape != x_skip_connection_2.shape:
            x_skip_connection_2 = TF.resize(x_skip_connection_2, x.shape[2], antialias=None)
        
        x = torch.cat([x, x_skip_connection_2], dim=1)
        x = self.up_double_convolution_7(x)
        x = self.upsample_8(x)

        if x.shape!= x_skip_connection_1.shape:
            x_skip_connection_1 = TF.resize(x_skip_connection_1, x.shape[2], antialias=None)

        x = torch.cat([x, x_skip_connection_1], dim=1)
        x = self.up_double_convolution_8(x)
        x = self.final_conv(x)

        print(x.shape)

        return x

            
if __name__ == '__main__':
    x_input = torch.rand((3, 572, 572))
    model = UNet()
    y = model(x_input.unsqueeze(0))