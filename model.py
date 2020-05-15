#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 14 15:04:06 2020

@author: ikyathvarmadantuluri
"""


import pandas as pd
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from sklearn.ensemble import RandomForestClassifier
import pickle


train = pd.read_csv('Data/cardio_train.csv',sep=';')


train = train.drop(train[(train.height<147) | (train.height>184)].index)
train = train.drop(train[(train.weight<48) | (train.weight>117)].index)

train.drop_duplicates(inplace=True)


s_list = ["age", "height", "weight", "ap_hi", "ap_lo"]
def standartization(x):
    x_std = x.copy(deep=True)
    for column in s_list:
        x_std[column] = (x_std[column]-x_std[column].mean())/x_std[column].std()
    return x_std 
train_std=standartization(train)

train=train_std


features=["age", "height", "weight",'ap_hi','ap_lo','smoke','alco','active','cholesterol','gluc']

train.astype(float)

X_train, X_test, y_train, y_test = train_test_split(train[features], train['cardio'], test_size=0.33, random_state=42)

RFModel = RandomForestClassifier()


RFModel.fit(X_train,y_train)


y_pred_xgb = RFModel.predict(X_test)

accuracy = RFModel.score(X_test, y_test)

pickle.dump(RFModel,open('model.pkl','wb'))




