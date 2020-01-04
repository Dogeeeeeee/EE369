import numpy as np
from sklearn.externals import joblib
import os
import pandas as pd

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
ans = myrfmodel.predict_proba(x_test)[:,1]
df = pd.read_csv('result.csv',encoding='utf-8')
for i in range(len(ans)):
    df['Predicted'].loc[i] = ans[i]
df.to_csv('submission.csv',encoding='utf-8',index= 0)
