import torch
import torch.nn as nn
import torch.nn.functional as F
from models.basicconv import BasicConv2d
class Stem(nn.Module):
    def __init__(self, in_channels):
        super(Stem, self).__init__()
        # block1
        self.conv1_1 = BasicConv2d(in_channels, 32, kernel_size=3, stride=2)
        self.conv1_2 = BasicConv2d(32, 32, kernel_size=3)
        self.conv1_3 = BasicConv2d(32, 64, kernel_size=3, padding=1)
        self.maxp1_1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv1_4 = BasicConv2d(64, 96, kernel_size=3, stride=2)
        # self.icp1 = InceptionA(96)
        # block2
        self.conv2_1_1 = BasicConv2d(160, 64, kernel_size=1)
        self.conv2_1_2 = BasicConv2d(64, 96, kernel_size=3)
        self.conv2_2_1 = BasicConv2d(160, 64, kernel_size=1)
        self.conv2_2_2 = BasicConv2d(64, 64, kernel_size=(7, 1), padding=1)
        self.conv2_2_3 = BasicConv2d(64, 64, kernel_size=(1, 7), padding=1)
        self.conv2_2_4 = BasicConv2d(64, 96, kernel_size=3, padding=1)
        # out
        self.maxp2 = nn.MaxPool2d(kernel_size=2, )
        self.conv3_1 = BasicConv2d(192, 192, kernel_size=3, stride=2, padding=1)
    def forward(self, x):
        # block1
        x1 = self.conv1_1(x)
        x1 = self.conv1_2(x1)
        x1 = self.conv1_3(x1)
        x1_1 = self.maxp1_1(x1)
        x1_2 = self.conv1_4(x1)
        x1 = [x1_1, x1_2]
        x1 = torch.cat(x1, dim=1)
        # block2
        x2_1 = self.conv2_1_2(self.conv2_1_1(x1))
        x2_2 = self.conv2_2_4(self.conv2_2_3(self.conv2_2_2(self.conv2_2_1(x1))))
        x2 = [x2_1, x2_2]
        x2 = torch.cat(x2, dim=1)
        # out
        x3_1 = self.maxp2(x2)
        x3_2 = self.conv3_1(x2)
        x3 = [x3_1, x3_2]
        x3 = torch.cat(x3, dim=1)
        return x3

class InceptionA(nn.Module):
    def __init__(self,in_channels):
        super(InceptionA,self).__init__()
        self.branch1 = BasicConv2d(in_channels,96,kernel_size=1)
        self.branch2_1 = BasicConv2d(in_channels,64,kernel_size=1)
        self.branch2_2 = BasicConv2d(64,96,kernel_size=5,padding=2)
        self.branch3_1 = BasicConv2d(in_channels,64,kernel_size=1)
        self.branch3_2 = BasicConv2d(64,96,kernel_size=3,padding=1)
        self.branch3_3 = BasicConv2d(96,96,kernel_size=3,padding=1)
        self.branch4_1 = nn.AvgPool2d(3,stride=1,padding=1)
        self.branch4_2 = BasicConv2d(in_channels,96,kernel_size=1)

    def forward(self,x):
        branchx1 = self.branch1(x)
        branchx2 = self.branch2_2(self.branch2_1(x))
        branchx3 = self.branch3_3(self.branch3_2(self.branch3_1(x)))
        branchx4 = self.branch4_2(self.branch4_1(x))
        outputs = [branchx1,branchx2,branchx3,branchx4]
        #print(branchx1.size(),branchx2.size(),branchx3.size(),branchx4.size())
        #print(torch.cat(outputs,dim=1).size())
        return torch.cat(outputs,dim=1)


class ReductionA(nn.Module):
    def __init__(self,in_channels):
        super(ReductionA,self).__init__()
        self.branch1 = nn.MaxPool2d(kernel_size=3,stride=2)
        self.branch2 = BasicConv2d(in_channels,in_channels,kernel_size=3,stride=2)
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels,in_channels,kernel_size=1),
            BasicConv2d(in_channels,in_channels,kernel_size=3,padding=1),
            BasicConv2d(in_channels,in_channels,kernel_size=3,stride=2)
        )
    def forward(self,x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = [x1,x2,x3]
        out = torch.cat(out,dim=1)
        return out



class InceptionB(nn.Module):
    def __init__(self,in_channels):
        super(InceptionB,self).__init__()
        self.branch1 =nn.Sequential(
            nn.AvgPool2d(3,stride=1,padding=1),
            nn.Conv2d(in_channels,128,kernel_size=1))

        self.branch2 =nn.Sequential(BasicConv2d(in_channels,384,kernel_size=1))
        self.branch3 = nn.Sequential(BasicConv2d(in_channels,192,kernel_size=1,padding=1),
                                     BasicConv2d(192,224,kernel_size=(1,7),padding=1),
                                     BasicConv2d(224,256,kernel_size=(7,1),padding=1))
        self.branch4 = nn.Sequential(
            BasicConv2d(in_channels,192,kernel_size=1,padding=2),
            BasicConv2d(192,192,kernel_size=(1,7),padding=1),
            BasicConv2d(192,224,kernel_size=(7,1),padding=1),
            BasicConv2d(224, 224, kernel_size=(1, 7),padding=1),
            BasicConv2d(224, 256, kernel_size=(7, 1),padding=1),
        )

    def forward(self,x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x4 = self.branch4(x)
        out = [x1,x2,x3,x4]
        out = torch.cat(out,dim=1)
        return out


class ReductionB(nn.Module):
    def __init__(self,in_channels):
        super(ReductionB,self).__init__()
        self.branch1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3,stride=2)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels,192,kernel_size=1),
            BasicConv2d(192,192,kernel_size=3,stride=2)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels,256,kernel_size=1),
            BasicConv2d(256,256,kernel_size=(1,7),padding=1),
            BasicConv2d(256,320,kernel_size=(7,1),padding=1),
            BasicConv2d(320,320,kernel_size=3,stride=2,padding=1)
        )
    def forward(self,x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = [x1,x2,x3]
        out = torch.cat(out,dim=1)
        return out


class InceptionC(nn.Module):
    def __init__(self,in_channels):
        super(InceptionC,self).__init__()
        self.branch1 = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1),
            BasicConv2d(in_channels, 128, kernel_size=1)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels,256,kernel_size=1)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels,384,kernel_size=1),
        )
        self.branch3_1 = BasicConv2d(384,236,kernel_size=(1,3),padding=(0,1))
        self.branch3_2 = BasicConv2d(384,256,kernel_size=(3,1),padding=(1,0))
        self.branch4 = nn.Sequential(
            BasicConv2d(in_channels,384,kernel_size=1,padding=1),
            BasicConv2d(384,448,kernel_size=(1,3),),
            BasicConv2d(448,512,kernel_size=(3,1)),
        )
        self.branch4_1 = BasicConv2d(512,256,kernel_size=(3,1),padding=(1,0))
        self.branch4_2 = BasicConv2d(512,256,kernel_size=(1,3),padding=(0,1))

    def forward(self,x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x3_1 = self.branch3_1(x3)
        x3_2 = self.branch3_2(x3)
        x3 = torch.cat([x3_1,x3_2],dim=1)
        x4 = self.branch4(x)
        x4_1 = self.branch4_1(x4)
        x4_2 = self.branch4_2(x4)
        x4 = torch.cat([x4_1,x4_2],dim=1)
        out = torch.cat([x1,x2,x3,x4],dim=1)
        return out

class Activate(nn.Module):
    def __init__(self,in_channels):
        super(Activate,self).__init__()
        self.branch1 = BasicConv2d(in_channels,96,kernel_size=1)
        self.branch2_1 = BasicConv2d(in_channels,64,kernel_size=1)
        self.branch2_2 = BasicConv2d(64,96,kernel_size=5,padding=2)
        self.branch3_1 = BasicConv2d(in_channels,64,kernel_size=1)
        self.branch3_2 = BasicConv2d(64,96,kernel_size=3,padding=1)
        self.branch3_3 = BasicConv2d(96,96,kernel_size=3,padding=1)
        self.branch4_1 = nn.AvgPool2d(3,stride=1,padding=1)
        self.branch4_2 = BasicConv2d(in_channels,96,kernel_size=1)

    def forward(self,x):
        branchx1 = self.branch1(x)
        branchx2 = self.branch2_2(self.branch2_1(x))
        branchx3 = self.branch3_3(self.branch3_2(self.branch3_1(x)))
        branchx4 = self.branch4_2(self.branch4_1(x))
        outputs = [branchx1,branchx2,branchx3,branchx4]
        #print(branchx1.size(),branchx2.size(),branchx3.size(),branchx4.size())
        #print(torch.cat(outputs,dim=1).size())
        return torch.cat(outputs,dim=1)

if __name__ == '__main__':
    x=torch.zeros(32,1388,1,1) #32, 1152, 11, 11
    net = Activate(x.shape[1])
    a=net(x)
    print(net(x).size())
