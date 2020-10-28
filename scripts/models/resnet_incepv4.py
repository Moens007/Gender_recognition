import random

from models.inceptions import InceptionA,ReductionA,InceptionB,ReductionB,InceptionC,Stem,Activate
import torch
import torch.nn as nn
import torch.nn.functional as F

'''class Bottlenneck(nn.Module):
    def __init__(self,in_channels,out_channels,stride=1,dowmsample=None):
        super(Bottlenneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels,out_channels,kernel_size=1,bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)  #批归一化
        self.conv2 = nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=stride,padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels,out_channels*4,kernel_size=1,bias=False)
        self.bn3 =  nn.BatchNorm2d(out_channels*4)
        self.downsample = dowmsample
        self.stride = stride
    def forward(self,x):
        residule = x
        out = self.bn1(self.conv1(x))
        out = self.bn2(self.conv2(out))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            residule = self.downsample(x)
        out += residule
        out = F.relu(out)
        return out'''

class Basicconv(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride,padding=(0,0)):
        super(Basicconv, self).__init__()
        self.conv = nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding)
        self.groBN = nn.GroupNorm(1,out_channels)
        self.relu = nn.ReLU()

    def forward(self,x):
        x = self.conv(x)
        x = self.groBN(x)
        x = self.relu(x)
        return x
class Mixed_3a(nn.Module):

    def __init__(self):
        super(Mixed_3a, self).__init__()
        self.maxpool = nn.MaxPool2d(3, stride=2)
        self.conv = Basicconv(64, 96, kernel_size=3, stride=2)

    def forward(self, x):
        x0 = self.maxpool(x)
        x1 = self.conv(x)
        out = torch.cat((x0, x1), 1)
        return out


class Mixed_4a(nn.Module):

    def __init__(self):
        super(Mixed_4a, self).__init__()

        self.branch0 = nn.Sequential(
            Basicconv(160, 64, kernel_size=1, stride=1),
            Basicconv(64, 96, kernel_size=3, stride=1)
        )

        self.branch1 = nn.Sequential(
            Basicconv(160, 64, kernel_size=1, stride=1),
            Basicconv(64, 64, kernel_size=(1,7), stride=1, padding=(0,3)),
            Basicconv(64, 64, kernel_size=(7,1), stride=1, padding=(3,0)),
            Basicconv(64, 96, kernel_size=(3,3), stride=1)
        )

class Resnet_incp_v4(nn.Module):
    def __init__(self):
        super(Resnet_incp_v4, self).__init__()
        self.feature = nn.Sequential(
            Stem(3),
            nn.Sequential(
            InceptionA(384),
            InceptionA(384),InceptionA(384),InceptionA(384),
                                   ),
            ReductionA(384),
            nn.Sequential(
                InceptionB(1152), InceptionB(1024), InceptionB(1024),
                InceptionB(1024), InceptionB(1024), InceptionB(1024), InceptionB(1024)

            ),
            ReductionB(1024),
            nn.Sequential(InceptionC(1536),
                          InceptionC(1388), InceptionC(1388)
                          ),
        )

        self.avg = nn.AvgPool2d(5)
        self.max = nn.MaxPool2d(5)

        self.fc1 = nn.Linear(1388,384)
        self.act = Activate(1388)
        self.fc2 = nn.Linear(384,2)
        # self.init_params()

    def forward(self,x):
        batch_size = len(x)
        x = self.feature(x)   ##32,1388,1,1
        x_avg  = self.avg(x)
        x_max = self.max(x)
        x = x_avg+x_max
        x = F.dropout(x,p=0.2)
        x1 = x.view(batch_size,-1)   #32,1388
        x1 = F.relu(self.fc1(x1))  #32,384
        x2 = self.act(x)*0.1
        x2 = x2.view(batch_size,-1)
        out = (x2 + x1)
        out = self.fc2(F.relu(out))
        return  out

    # def init_params(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d):
    #             torch.nn.init.xavier_uniform_(m.weight)
    #         elif isinstance(m, nn.BatchNorm2d):
    #             torch.nn.init.constant_(m.weight, 1)
    #             torch.nn.init.constant_(m.bias, 0)
    #         elif isinstance(m, nn.Linear):
    #             torch.nn.init.normal_(m.weight, std=1e-3)


if __name__ == '__main__':
    random.seed(32)
    x = torch.randn(32,3,200,200 ) #x1:[32, 256]
    net = Resnet_incp_v4()
    print(net(x).size())