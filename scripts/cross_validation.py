import numpy as np
from sklearn.model_selection import  KFold
import pandas as pd
import random

'''测试数据'''
random.seed(32)
# df = pd.read_csv('../dataset/train.csv').values
# rd = random.sample(range(len(df)),len(df))
# df = df[rd]
# df = pd.DataFrame({'id':df[:,0],'label':df[:,1]})
# df.to_csv('../dataset/train_corss_validation',index=False)


df = pd.read_csv('../dataset/train_corss_validation').values
id,labels = df[:,0],df[:,1]
data_size = len(id)
def get_kfole(k=5):
    kf = KFold(k)
    fold_num = 0
    for train_index,test_index in kf.split(id):
        train_id,test_id = id[train_index],id[test_index]
        train_label,test_label = labels[train_index],labels[test_index]
        pd.DataFrame({'id':train_id,'label':train_label}).to_csv('../dataset/five_fold/train_kfold_{}.csv'.format(fold_num),index=False)
        pd.DataFrame({'id':test_id,'label':test_label}).to_csv('../dataset/five_fold/test_kfold_{}.csv'.format(fold_num),index=False)
        fold_num += 1

if __name__ == '__main__':
    get_kfole()


