# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 08:03:27 2021

@author: mashituo
"""
import pandas as pd
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.metrics import confusion_matrix,accuracy_score
df=pd.read_csv("wine.csv")
y=df['class']
X=df[['x1','x2','x3','x4','x5','x6','x7','x8','x9','x10','x11','x12','x13']]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
RF=RandomForestClassifier()
Ab=AdaBoostClassifier()
RF.fit(X_train,y_train)
y_pred1=RF.predict(X_test)
Ab.fit(X_train,y_train)
y_pred2=Ab.predict(X_test)
print(confusion_matrix(y_test,y_pred1))
print(accuracy_score(y_test,y_pred1))
print(confusion_matrix(y_test,y_pred2))
print(accuracy_score(y_test,y_pred2))

