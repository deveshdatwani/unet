import torch
from torch import nn
torch.set_grad_enabled(True) 
from torch.autograd import Variable



class DiceLoss(nn.Module):
    
    def __init__(self, smooth=0.01):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.loss = nn.CrossEntropyLoss()


    def forward(self, prediction, target):
        prediction_flat = prediction.view(-1)
        target_flat = target.view(-1)
        intersection = (prediction_flat * target_flat).sum()
        prediction_sum  = (prediction_flat * prediction_flat).sum()
        target_sum = target_flat.sum()
        loss = 1 - (((2 * intersection) / (prediction_sum + target_sum)) / 2)   

        return loss