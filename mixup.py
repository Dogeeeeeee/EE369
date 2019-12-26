import numpy as np
import random
import os
import csv


# 读取数据
y_train_path = './train_val.csv'
x_train_path = './train_val/'
labels = []
y_train_val = np.genfromtxt(y_train_path, delimiter=',', skip_header=1, usecols=1)
###############################
file_all = os.listdir(x_train_path)
file_all.sort()
idx = 0
flag = False
while(idx<500):
    first = random.randint(0,464)
    label = y_train_val[first]
    second = random.randint(0,464)
    while(y_train_val[second] != label):
        second = random.randint(0, 464)

    tmp1 = np.load(x_train_path+file_all[first])
    myarray1 = np.array(tmp1['voxel'])
    mymask1 = np.array(tmp1['seg'])

    tmp2 = np.load(x_train_path+file_all[second])
    myarray2 = np.array(tmp2['voxel'])
    mymask2  =np.array(tmp2['seg'])

    lambda1 = random.randint(1,99)/100.0
    myarray = lambda1*myarray1+(1-lambda1)*myarray2
    mymask = lambda1*mymask1+(1-lambda1)*mymask2

    np.savez('./train_val/mixup/mixup'+str(idx)+'.npy',voxel=myarray,seg = mymask)
    labels.append([label])
    idx = idx+1
np.savetxt('mixuplabel.txt',labels)

###############)#########################
