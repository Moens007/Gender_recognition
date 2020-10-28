import pandas as pd
import matplotlib.pyplot as plt
import random
import os
import shutil
import numpy as np
random.seed(32)
df = pd.read_csv("../dataset/train.csv", encoding="utf8").values

# if not os.path.exists('../dataset//train_train'):
#     os.mkdir('../dataset//train_train')
# if not os.path.exists('../dataset//train_test'):
#     os.mkdir('../dataset//train_test')

# save_train_address = '../dataset//train_train'
# save_test_address = '../dataset//train_test'
out_path = '../dataset/train'

#打乱df
rd = random.sample(range(len(df)),len(df))
df = df[rd]


#比例8比2
train_num = int(len(df)*0.8)
test_num = int(len(df)*0.2)

train_files = df[:train_num]
test_files = df[train_num:len(df)]

# for img,labels in train_files:
#     shutil.copy(os.path.join(out_path,str(img)+'.jpg'),os.path.join(save_train_address,str(img)+'.jpg'))
#
# for img,labels in test_files:
#     shutil.copy(os.path.join(out_path,str(img)+'.jpg'),os.path.join(save_test_address,str(img)+'.jpg'))
'''数据分离'''
train_data = {'id':train_files[:,0],'label':train_files[:,1]}
test_data = {'id':test_files[:,0],'label':test_files[:,1]}

df_train = pd.DataFrame(train_data)
df_test = pd.DataFrame(test_data)
#df_train.to_csv('../dataset//train_train.csv',index=False)
#df_test.to_csv('../dataset//train_test.csv',index=False)

plt.figure()
plt.subplot(121)
plt.title('train')
df_test['label'].value_counts().plot(kind='bar')
plt.subplot(122)
plt.title('test')
df_train['label'].value_counts().plot(kind='bar')
plt.show()