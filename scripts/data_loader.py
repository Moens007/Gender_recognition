import pandas as pd
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
import cv2
from get_albumentation import get_train_transform
import numpy as np
import torch
'''#数据提取'''
class TrainDataset(Dataset):
    def __init__(self, filepath, data_folder, transform=None):
        self.data = pd.read_csv(filepath, encoding='utf8').values[:]
        self.data_folder = data_folder
        self.transform = transform
        self.x_data=self.data[:, 0]
        self.y_data=self.data[:, 1]

    def __getitem__(self, index):
        x, y = self.x_data[index], self.y_data[index]
        img = cv2.imread(os.path.join(self.data_folder, str(x) + '.jpg'),1)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        if self.transform:
            img = self.transform(image=img)['image']
        img = np.transpose(img)
        img = torch.from_numpy(img).type(torch.FloatTensor)
        return img, y

    def __len__(self, ):
         return len(self.data)
        # return 1000


class TestDataset(Dataset):
    def __init__(self,filepath,data_folder,transform=None):
        self.data = pd.read_csv(filepath,encoding='utf8').values
        self.data_folder = data_folder
        self.transform = transform
        self.x_data = self.data[:,0]

    def __getitem__(self, index):
        x = self.x_data[index]
        img = cv2.imread(os.path.join(self.data_folder, str(x) + '.jpg'), 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform:
            img = self.transform(image=img)['image']
        img = np.transpose(img)
        img = torch.from_numpy(img).type(torch.FloatTensor)
        return img

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    transforms = get_train_transform()
    dataset = TrainDataset('../dataset//train.csv','../dataset//train/',transforms)
    dataloader = DataLoader(dataset,batch_size=32,shuffle=False)
    for i,img in enumerate(dataloader,0):
        print(img)
    # dataset = TrainDataset('../dataset//train_train.csv', '../dataset//train/', transforms.ToTensor())
    # dataloader = DataLoader(dataset,batch_size=32,shuffle=False)
    # for img,label in dataloader:
    #     pass



