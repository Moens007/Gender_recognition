from torchvision import models
import torch.nn as nn
import torch
import torch.nn.functional as F
class Resnet_101(nn.Module):
    def __init__(self):
        super(Resnet_101, self).__init__()
        self.model1 = models.resnet101()
        self.fc1 = nn.Linear(1000,64)
        self.fc2 = nn.Linear(64,2)
    def forward(self,x):
        x = self.model1(x)
        x = self.fc2(F.relu(self.fc1(x)))
        return x

if __name__ == '__main__':
    x = torch.randn(32,3,200,200)
    net = Resnet_101()
    print(net(x).size())