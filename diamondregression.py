# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 23:15:52 2022

@author: chakr
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from feature_engine.encoding import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score

df = pd.read_csv('diamonds.csv')
df.info()
enc = OneHotEncoder(drop_last=True,top_categories = None,variables = ['cut','color','clarity'])
databq = enc.fit_transform(df)
databq.info()
databq.rename(columns = {'cut_Very Good':'cut_Excl'},inplace = True)
databqcols = databq.columns.to_list()
scalar = MinMaxScaler()
databq[['carat','depth','table','price','x','y','z']] = scalar.fit_transform(databq[['carat','depth','table','price','x','y','z']])

y = databq['price']
X = databq.drop(['price'],axis = 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

model = LinearRegression()
output = model.fit(X_train,y_train) 

test_prediction = output.predict(X_test)

R2_value = r2_score(y_test,test_prediction)




