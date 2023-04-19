from torch import nn
import torch


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3)
        self.conv2 = nn.Conv2d(64, 64, 3)
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.conv4 = nn.Conv2d(128, 128, 3)
        self.conv5 = nn.Conv2d(128, 256, 3)
        self.conv6 = nn.Conv2d(256, 256, 3)
        self.conv7 = nn.Conv2d(256, 512, 3)
        self.conv8 = nn.Conv2d(512, 512, 3)
        self.conv9 = nn.Conv2d(512, 1024, 3)
        self.conv10 = nn.Conv2d(1024, 1024, 3)
        self.conv11 = nn.Conv2d(1024, 512, 3)
        self.conv12 = nn.Conv2d(512, 512, 3)
        self.conv13 = nn.Conv2d(512, 256, 3)
        self.conv14 = nn.Conv2d(256, 256, 3)
        self.conv15 = nn.Conv2d(256, 128, 3)
        self.conv16 = nn.Conv2d(128, 128, 3)
        self.conv17 = nn.Conv2d(128, 64, 3)
        self.conv18 = nn.Conv2d(64, 64, 3)
        self.conv19 = nn.Conv2d(64, 2, 1)
        self.up1 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.up2 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.up4 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.max1 = nn.MaxPool2d(2)
        self.relu = nn.ReLU()


    def forward(self, x):
        x = self.relu(self.conv1(x))
        x1 = self.relu(self.conv2(x))
        x = self.max1(x1)
        x = self.relu(self.conv3(x))
        x2 = self.relu(self.conv4(x))
        x = self.max1(x2)
        x = self.relu(self.conv5(x))
        x3 = self.relu(self.conv6(x))
        x = self.max1(x3)
        x = self.relu(self.conv7(x))
        x4 = self.relu(self.conv8(x))
        x = self.max1(x4)
        x = self.relu(self.conv9(x))
        x = self.relu(self.conv10(x))
        x = self.up1(x)
        x = torch.concatenate((x4[:,4:-4,4:-4], x), dim=0)
        x = self.relu(self.conv11(x))
        x = self.relu(self.conv12(x))
        x = self.up2(x)
        x = torch.concatenate((x3[:,16:-16,16:-16], x), dim=0)
        x = self.relu(self.conv13(x))
        x = self.relu(self.conv14(x))
        x = self.up3(x)
        x = torch.concatenate((x2[:,40:-40,40:-40], x), dim=0)
        x = self.relu(self.conv15(x))
        x = self.relu(self.conv16(x))
        x = self.up4(x)
        x = torch.concatenate((x1[:,88:-88,88:-88], x), dim=0)
        x = self.relu(self.conv17(x))
        x = self.relu(self.conv18(x))
        x = self.relu(self.conv19(x))

        return x
    


model = UNet()
inPut = torch.rand((3,572,572))
y = model(inPut)
print(y.shape)

        
