# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 09:22:07 2021

@author: mashituo
"""
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix,accuracy_score
df=pd.read_csv("soybean-large.csv")
df=df.replace('?',-1)
df=df.dropna()
X=df.iloc[:,1:35]
y=df.label
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
svm=SVC()
svm.fit(X_train,y_train)
y_predict=svm.predict(X_test)
#evaluate
print(confusion_matrix(y_test,y_predict))
print(accuracy_score(y_test,y_predict))


