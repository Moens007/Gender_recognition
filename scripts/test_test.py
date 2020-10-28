from tqdm import tqdm

from scripts.models.resnet_incepv4 import Resnet_incp_v4
import torch
import pandas as pd
from scripts.data_loader import TestDataset
from torchvision import transforms
from torch.utils.data import DataLoader
from scripts.models.resnet_50 import Resnet_50
from scripts.models.resnet_101 import Resnet_101
import torch.nn as nn
import numpy as np

'''测试数据准确率'''
transform = transforms.Compose([
    transforms.ToTensor(),
])
dataset = TestDataset('../dataset//test.csv','../dataset//test/',transform)
data_load = DataLoader(dataset,batch_size=16,shuffle=False)

model = Resnet_incp_v4()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
model.load_state_dict(torch.load("../runs/exp12/best_checkpoint_ep40.pth")['state_dict'])
model.eval()#修改


predict_total = []
for img in tqdm(data_load):
    img = img.to(device)
    predict = model(img)
    _,predict_y = torch.max(predict.data,dim=1)
    predict_total.extend(list(predict_y.cpu().numpy()))

# out = {'lable':predict_total}
# out = pd.DataFrame(out)
# #outdata = out.to_csv('../dataset/test_out.csv',encoding='utf8',index=False)

out1 = pd.read_csv('../dataset/test.csv',encoding='utf8').values
out = {'id':out1[:,0],'label':predict_total}
out = pd.DataFrame(out)
outdata = out.to_csv('../dataset/output_exp/output_exp12.csv',index=False)
