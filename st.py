import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import  metrics
from kymatio import HarmonicScattering3D as st
import torch
import os


# 读取数据
y_train_path = './train_val.csv'
x_train_path = './train_val/'
x_test_path = './test/'

x_train_store_path = './sttrain_val'
SIZE = 16
scattering = st(J=2, shape=(2*SIZE, 2*SIZE, 2*SIZE))

y_train_val = np.genfromtxt(y_train_path, delimiter=',', skip_header=1, usecols=1)
###############################
file_all = os.listdir(x_train_path)
file_all.sort()
flag = False
for file in file_all:
    tmp = np.load(x_train_path+file)
    myarray = np.array(tmp['voxel'])[50-SIZE:50+SIZE,50-SIZE:50+SIZE,50-SIZE:50+SIZE]
    mymask  =np.array(tmp['seg'])[50-SIZE:50+SIZE,50-SIZE:50+SIZE,50-SIZE:50+SIZE]
   # img = myarray[40:60, 40:60, 40:60]
    img = myarray*mymask
    img = img[np.newaxis,:,:,:]/ 255.0
    img =torch.from_numpy(img)
    img = scattering(img).numpy()
    img = img[0,:,:,:,:,:]
    arr = img[0,0,:,:,:]
    arr = arr[np.newaxis,:,:,:]
    for i in range(6):
        for j in range(4):
            if(i==0 and j==0):
                continue
            tmp = img[i,j,:,:,:]
            tmp = tmp[np.newaxis,:,:,:]
            arr = np.vstack((arr,tmp))
    np.save('./sttrain_val/'+file,arr)

###############)#########################
file_all = os.listdir(x_test_path)
file_all.sort()
flag = False
for file in file_all:
    tmp = np.load(x_test_path + file)
    myarray = np.array(tmp['voxel'])[50-SIZE:50+SIZE,50-SIZE:50+SIZE,50-SIZE:50+SIZE]
    mymask = np.array(tmp['seg'])[50-SIZE:50+SIZE,50-SIZE:50+SIZE,50-SIZE:50+SIZE]
    img = myarray * mymask
    img = img[np.newaxis, :, :, :] / 255.0
    img = torch.from_numpy(img)
    img = scattering(img).numpy()
    img = img[0, :, :, :, :, :]

    arr = img[0, 0, :, :, :]
    arr = arr[np.newaxis, :, :, :]
    for i in range(6):
        for j in range(4):
            if (i == 0 and j == 0):
                continue
            tmp = img[i, j, :, :, :]
            tmp = tmp[np.newaxis, :, :, :]
            arr = np.vstack((arr, tmp))
    np.save('./sttest/' + file, arr)
###########################################