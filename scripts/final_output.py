import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

'''投票输出'''
test_sample = pd.read_csv('../dataset/test.csv').values

pth = os.path.dirname(__file__)
out_path = os.path.join(pth,'../dataset/five_fold/data_out')
out_list = os.listdir(out_path)

data = np.zeros([test_sample.shape[0],1])
for i,file_name in enumerate(out_list):
    if i == 0:
        data = pd.read_csv(os.path.join(out_path,file_name)).values[:,1]
    else:
        out_put = pd.read_csv(os.path.join(out_path,file_name)).values[:,1]
        data = np.dstack([data,out_put])

data = data.reshape(-1,len(out_list))
out = []
for i in data:
    out.append(np.argmax(np.bincount(i)))

out = pd.DataFrame({'id':test_sample[:,0],'label':out})

out['label'].value_counts().plot(kind='bar')
plt.title('label')
plt.show()

out.to_csv('../dataset/final_out.csv',index=False)



#
# data1 = pd.read_csv('../dataset/output_exp/output_exp2.csv').values[:,1]
# data2 = pd.read_csv('../dataset/output_exp/output_exp3.csv').values[:,1]
# data3 = pd.read_csv('../dataset/output_exp/output_exp4.csv').values[:,1]
# data4 = pd.read_csv('../dataset/output_exp/output_exp5.csv').values[:,1]
# data5 = pd.read_csv('../dataset/output_exp/output_exp6.csv').values[:,1]
# data6 = pd.read_csv('../dataset/output_exp/output_exp7.csv').values[:,1]
# data7 = pd.read_csv('../dataset/output_exp/output_exp8.csv').values[:,1]
#
#
# #选取最大
# data = np.stack([data1,data2,data3,data4,data5,data6,data7],1)
#
#
# out1 = []
# for i in data:
#     out1.append(np.argmax(np.bincount(i)))
# final_out = pd.DataFrame({'id':test_sample[:,0],'label':out})
# final_out.to_csv('../dataset/final_out.csv',index=False)
#
# final_out['label'].value_counts().plot(kind='bar')
# plt.title('label')
# plt.show()