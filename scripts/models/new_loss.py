import torch
from scripts.utils import Config
import torch.nn as nn
import torch.nn.functional as F
import numpy
import random
cfg = Config()
random.seed(32)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class new_loss(nn.Module):
    def __init__(self,first = nn.CrossEntropyLoss(),second = nn.SoftMarginLoss()):
        super(new_loss,self).__init__()
        self.first = first
        self.second = second
        self.fc2 = nn.Linear(2,1)
        self.sig = nn.Sigmoid()

    def forward(self, logits, labels):
        t = 0.5
        len = logits.shape[0]
        loss_1 = self.first(logits,labels)
        label_y = ( labels*2 - 1)
        loss_2 = 0
        logits_y = self.fc2(logits)
        for i in range(len):
            loss_2 += self.second(logits_y[i],label_y[i])
        loss = loss_1 + t*loss_2
        return loss/2

if __name__ == '__main__':
    x = torch.randn(13,2)
    y = torch.LongTensor(numpy.random.randint(0,2,[13]))
    loss = new_loss()
    loss1 = loss(x,y)
    print(loss1)