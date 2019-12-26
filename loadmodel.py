import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import  metrics
from kymatio import HarmonicScattering3D as st
import torch
from sklearn.externals import joblib
import os

x_test_path = './sttest/'

file_all = os.listdir(x_test_path)
file_all.sort()
flag = False
x_test = []
for file in file_all:
    tmp = np.load(x_test_path + file)
    arrTmp = []
    for i in range(24):
        arrTmp.append(
            (tmp[i, :, :, :].mean(), tmp[i, :, :, :].std(), tmp[i, :, :, :].var(), np.median(tmp[i, :, :, :])))
    arrTmp = np.array(arrTmp)
    arrTmp = arrTmp.flatten()
    arrTmp = arrTmp.tolist()
    x_test.append(arrTmp)


myrfmodel = joblib.load('rf.pkl')

print(myrfmodel.predict_proba(x_test)[:, 1])
