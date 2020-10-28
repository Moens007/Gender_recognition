from torch import nn
import  torch

class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, bias=False) # verify bias false
        #
        self.bn = nn.BatchNorm2d(out_channels,
                                 eps=0.001, # value found in tensorflow
                                 momentum=0.1, # default pytorch value
                                 affine=True
                               )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

if __name__ == '__main__':
    net = BasicConv2d(3,64,3)
    x = torch.randn(32,3,200,200)
    print(net(x).size())