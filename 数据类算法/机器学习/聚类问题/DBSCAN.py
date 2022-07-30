# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 14:03:27 2021

@author: mashituo
"""
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import pandas as pd
df=pd.read_csv("Live.csv")
X=df[['num_reactions','num_comments','num_shares','num_likes','num_loves','num_wows','num_hahas','num_sads','num_angrys']]
labels=DBSCAN(eps=0.1,min_samples=5).fit(X)
# 获取簇心
# 获取归集后的样本所属簇对应值
y_kmean = labels.labels_
x1=df.num_wows
x2=df.num_hahas
plt.scatter(x1,x2,s=len(x1),c=y_kmean)
plt.show()