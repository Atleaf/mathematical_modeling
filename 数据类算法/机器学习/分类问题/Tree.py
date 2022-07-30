# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 09:11:21 2021

@author: mashituo
"""
import pandas as pd
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix,accuracy_score
import pydotplus
from IPython.display import Image,display
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
dtree=DecisionTreeClassifier(max_depth=5)
dtree.fit(X_train,y_train)
y_predict=dtree.predict(X_test)
#evaluate
print(confusion_matrix(y_test,y_predict))
print(accuracy_score(y_test,y_predict))
dot_data=tree.export_graphviz(dtree,
               out_file=None,
               feature_names=['price','comprice','tech','door','size','safety'],
               class_names=y,
               filled=True,
               rounded=True
               )
graph=pydotplus.graph_from_dot_data(dot_data)
display(Image(graph.create_png()))


