# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 08:57:58 2021

@author: mashituo
"""
import pandas as pd
from sklearn import naive_bayes
from sklearn.metrics import confusion_matrix,accuracy_score
df=pd.read_csv("car.csv")
df.replace("vhigh", 4, inplace=True)
df.replace("high", 3, inplace=True)
df.replace("med", 2, inplace=True)
df.replace("low", 1, inplace=True)
df.replace("big", 3, inplace=True)
df.replace("small", 1, inplace=True)
df.replace("more", 5, inplace=True)
df.replace("5more", 5, inplace=True)
df=df.dropna()
X=df[['price','comprice','tech','door','size','safety']]
y=df.acceptable
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
#train regressor
NB=naive_bayes.GaussianNB()
NB.fit(X_train,y_train)
y_predict=NB.predict(X_test)
#evaluate
print(confusion_matrix(y_test,y_predict))
print(accuracy_score(y_test,y_predict))

