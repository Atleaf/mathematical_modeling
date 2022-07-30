# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 14:03:27 2021

@author: mashituo
"""
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd
df=pd.read_csv("Live.csv")
X=df[['num_reactions','num_comments','num_shares','num_likes','num_loves','num_wows','num_hahas','num_sads','num_angrys']]
km=KMeans(n_clusters=5, init='k-means++', max_iter=30)
km.fit(X)
# 获取簇心
centroids = km.cluster_centers_
# 获取归集后的样本所属簇对应值
y_kmean = km.predict(X)
x1=df.num_wows
x2=df.num_hahas
plt.scatter(x1,x2,s=len(x1),c=y_kmean)
plt.show()