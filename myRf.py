import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import  metrics
from kymatio import HarmonicScattering3D as st
import torch
from sklearn.externals import joblib
import os

# 读取数据
y_train_path = './train_val.csv'
x_train_path = './sttrain_val/'
x_test_path = './sttest/'

y_train_val = np.genfromtxt(y_train_path, delimiter=',', skip_header=1, usecols=1)
y_test = np.genfromtxt(y_ground, delimiter=',', skip_header=1, usecols=1)
###############################
file_all = os.listdir(x_train_path)
file_all.sort()
x_train_val = []
for file in file_all:
    tmp = np.load(x_train_path+file)
    arrTmp = []
    for i in range(24):
        arrTmp.append((tmp[i,:,:,:].mean(),tmp[i,:,:,:].std(),tmp[i,:,:,:].var(),np.median(tmp[i,:,:,:])))
    arrTmp = np.array(arrTmp)
    arrTmp = arrTmp.flatten()
    arrTmp = arrTmp.tolist()
    x_train_val.append(arrTmp)
###############)#########################
file_all = os.listdir(x_test_path)
file_all.sort()
flag = False
x_test = []
for file in file_all:
    tmp = np.load(x_test_path+file)
    arrTmp = []
    for i in range(24):
        arrTmp.append((tmp[i,:,:,:].mean(),tmp[i,:,:,:].std(),tmp[i,:,:,:].var(),np.median(tmp[i,:,:,:])))
    arrTmp = np.array(arrTmp)
    arrTmp = arrTmp.flatten()
    arrTmp = arrTmp.tolist()
    x_test.append(arrTmp)
###########################################
x_train_val= np.array(x_train_val)
x_exam = np.array(x_test)

#x_train, x_test, y_train, y_test = train_test_split(x_train_val, y_train_val, test_size=0.3, random_state=12)

# Create the model with 100 trees
model = RandomForestClassifier(n_estimators=32,
                               max_features = 'auto',
                               criterion='entropy',
                               oob_score = True,
                               n_jobs=-1)
# Fit on training data
model.fit(x_train_val, y_train_val)


# Actual class predictions
rf_predictions = model.predict(x_exam)
# Probabilities for each class
rf_probs = model.predict_proba(x_exam)[:, 1]
np.savetxt('outputprob.txt',rf_probs)
np.savetxt('output.txt',rf_predictions)


joblib.dump(model,'rf.pkl')
clf=joblib.load('rf.pkl')


