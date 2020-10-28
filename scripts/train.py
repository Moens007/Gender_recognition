import torch.optim as optim
import torch.nn  as nn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import torch
from runx.logx import logx
import random
import os

from data_loader import TrainDataset
from models.resnet_incepv4 import Resnet_incp_v4
from models.resnet_50 import Resnet_50
from models.resnet_101 import Resnet_101
from scripts.utils import Config
from models.new_loss import new_loss
from utils import get_logdir
from get_albumentation import get_train_transform

'''训练模型'''
random.seed(32)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
cfg = Config()

# logx.initialize(get_logdir("../runs"), tensorboard=True, coolname=False)

train_transform = get_train_transform()
# transform 强化
# train_transform = transforms.Compose([
#     # transforms.RandomResizedCrop(150, scale=(0.08, 0.98), ratio=(0.75, 1.3333333333333333), interpolation=2),
# transforms.RandomApply([
#         transforms.RandomChoice([
#             transforms.ColorJitter(brightness=[0.8, 1.2], contrast=[0.8, 1.2], saturation=[0.8, 1.2], hue=[-0.2, 0.2]),
#             transforms.Compose([transforms.Resize((150, 150)), transforms.Resize((200, 200))]),  # 随机变形
#             transforms.Compose([transforms.RandomResizedCrop(150, scale=(0.78, 1), ratio=(0.90, 1.10), interpolation=2),
#                                 transforms.Resize((200, 200))]),  # 随机长宽比裁剪
#             transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转
#             transforms.RandomRotation((-15, 15), resample=False, expand=False, center=None),  # 随机旋转
#             ])], p=0.7),
#     transforms.ToTensor(),
# ])  #训练集数据增强

# test_transform = transforms.Compose([
#     transforms.ToTensor(),
# ])  #测试集数据

# 导入读取数据

# dataset_train = TrainDataset('../' + cfg.root_folder + '/five_fold/train_kfold_1.csv','../' + cfg.root_folder + '/train/', train_transform)
# train_loader = DataLoader(dataset_train, batch_size=cfg.bs, shuffle=True)
# test_data = TrainDataset('../' + cfg.root_folder + '/five_fold/test_kfold_1.csv','../' + cfg.root_folder + '/train/', )
# test_load = DataLoader(test_data, batch_size=cfg.bs, shuffle=False)

# 构建模型
print("training in", device)
model = Resnet_incp_v4()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=cfg.lr, )
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=4e-08)
model.load_state_dict(torch.load("..\\runs\exp10\last_checkpoint_ep0.pth")['state_dict'])  # warmup

model.to(device)


def train():
    for time in range(5):
        logx.initialize(get_logdir("../runs"), tensorboard=True, coolname=False)

        model.load_state_dict(torch.load("..\\runs\exp10\last_checkpoint_ep0.pth")['state_dict'])  # warmup

        dataset_train = TrainDataset('../' + cfg.root_folder + '/five_fold/train_kfold_{}.csv'.format(time),
                                     '../' + cfg.root_folder + '/train/', train_transform)
        train_loader = DataLoader(dataset_train, batch_size=cfg.bs, shuffle=True)
        test_data = TrainDataset('../' + cfg.root_folder + '/five_fold/test_kfold_{}.csv'.format(time),
                                 '../' + cfg.root_folder + '/train/', )
        test_load = DataLoader(test_data, batch_size=cfg.bs, shuffle=False)

        # train
        for epoch in range(cfg.epoch):
            loss_epoch = 0
            total = 0
            correct = 0
            for i, (x, y) in enumerate(train_loader, 1):
                x, y = x.to(device), y.to(device)
                y_hat = model(x)
                # 计算正确率
                total += x.size(0)
                _, predict = torch.max(y_hat.data, dim=1)
                correct += (predict == y).sum().item()

                # 损失
                loss = criterion(y_hat, y)
                loss_epoch += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # 过程可视化
                if i % 30 == 0:
                    print('epoch:%d,  enumerate:%d,  loss_avg:%f,  now_acc:%f' % (
                    epoch, i, loss_epoch / i, correct / total))

            # epoch matric 可视化
            train_loss = loss_epoch / i
            train_acc = (correct / total) * 100
            logx.metric('train', {'loss': train_loss, 'acc': train_acc}, epoch)

            # valid
            # 开发集正确率
            correct = 0
            total = 0
            val_loss = 0
            with torch.no_grad():
                for i, (img, label) in enumerate(test_load, 1):
                    img, label = img.to(device), label.to(device)
                    output = model(img)
                    loss = criterion(output, label)
                    val_loss += loss.cpu().item()
                    _, predicted = torch.max(output.data, dim=1)  # 最大值，位置
                    total += img.size(0)
                    correct += (predicted == label).sum().item()
            val_acc = (100 * correct / total)
            val_loss /= i
            logx.metric('val', {'loss': val_loss, 'acc': val_acc}, epoch)
            # epoch lossand other metric
            print('epoch over; train_loss:%f, val_loss:%f, train_acc=%f, val_acc:%f' % (
                train_loss, val_loss, train_acc, val_acc))
            logx.save_model({'state_dict': model.state_dict(), 'epoch': epoch}, val_acc, higher_better=True,
                            epoch=epoch, delete_old=True)
            scheduler.step()
train()
