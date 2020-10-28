import os
import torch
from  torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
from tqdm import tqdm
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

'''TTA  测试时间增强'''
from scripts.data_loader import TestDataset
from scripts.models.resnet_incepv4 import Resnet_incp_v4

with torch.no_grad():
    device = torch.device('cuda:0'if torch.cuda.is_available() else 'gpu')
    transform = transforms.Compose([transforms.ToTensor()])

    tfms =[transforms.Compose([transforms.ColorJitter(brightness=[0.8, 1.2], contrast=[0.8, 1.2], saturation=[0.8, 1.2], hue=[-0.2, 0.2]),transforms.ToTensor()]),
           transforms.Compose([transforms.ColorJitter(brightness=[0.8, 1.2], contrast=[0.8, 1.2], saturation=[0.8, 1.2],
                                                      hue=[-0.2, 0.2]), transforms.ToTensor()]),
           transforms.Compose([transforms.ColorJitter(brightness=[0.8, 1.2], contrast=[0.8, 1.2], saturation=[0.8, 1.2],
                                                      hue=[-0.2, 0.2]), transforms.ToTensor()]),
           transforms.Compose([transforms.Compose([transforms.Resize((150, 150)), transforms.Resize((200, 200))]), transforms.ToTensor()]),
           transforms.Compose([transforms.Compose([transforms.RandomResizedCrop(150, scale=(0.78, 1), ratio=(0.90, 1.10), interpolation=2),
                                    transforms.Resize((200, 200))]),transforms.ToTensor()]),
           transforms.Compose([transforms.Compose(
               [transforms.RandomResizedCrop(150, scale=(0.78, 1), ratio=(0.90, 1.10), interpolation=2),
                transforms.Resize((200, 200))]), transforms.ToTensor()]),
           transforms.Compose([transforms.Compose([transforms.RandomResizedCrop(150, scale=(0.78, 1), ratio=(0.90, 1.10), interpolation=2),
                                    transforms.Resize((200, 200))]), transforms.ToTensor()]),
           transforms.Compose([transforms.RandomHorizontalFlip(p=0.5), transforms.ToTensor()]),
           transforms.Compose([transforms.RandomRotation((-20, 20), resample=False, expand=False, center=None), transforms.ToTensor()]),
           transforms.Compose([transforms.RandomRotation((-20, 20), resample=False, expand=False, center=None),
                               transforms.ToTensor()]),
           ]

    model = Resnet_incp_v4()
    model.to(device)
    model.load_state_dict(torch.load('../runs/exp12/best_checkpoint_ep40.pth')['state_dict'])
    model.eval()

    test_sample = pd.read_csv('../dataset/test.csv').values
    pwd = os.getcwd()
    file_dir = os.path.join(pwd,'../dataset/test')
    file_name = pd.read_csv(os.path.join(pwd,'../dataset/test.csv')).values[:,0]
    predict_out = []

    for i in tqdm(range(file_name.shape[0])):
        predict_now = []
        img = Image.open(os.path.join(file_dir,str(file_name[i])+'.jpg'))
        for i in range(9):
            transform = tfms[i]
            img1 = transform(img)
            img1 = img1.view(1,3,200,200)
            img1 = img1.to(device)
            predict = model(img1)
            _,predict_y = torch.max(predict.data,dim=1)
            predict_now.extend(predict_y.cpu().numpy())
        predict_out.append(np.argmax(np.bincount(predict_now)))

    predict_out = pd.DataFrame({'id':test_sample[:,0],'label':predict_out})
    predict_out.to_csv('../dataset/five_fold/data_out/exp12.csv',index=False)











