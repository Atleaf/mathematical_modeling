# -*- coding: utf-8 -*-
"""
Created on Sat Jan 16 09:19:43 2021

@author: mashituo
"""
from factor_analyzer import FactorAnalyzer
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import bartlett
#参数设置
data=pd.read_excel('期末考试.xlsx')
data=data.set_index('姓名')
n_factors=2#因子数量
 
#用检验是否进行
corr=list(data.corr().to_numpy())
bartlett(*corr)
 
#开始计算
fa = FactorAnalyzer(n_factors=n_factors,method='principal',rotation="varimax")
fa.fit(data)
communalities= fa.get_communalities()#共性因子方差
loadings=fa.loadings_#成分矩阵，可以看出特征的归属因子
 
#画图
plt.figure()
ax = sns.heatmap(loadings, annot=True, cmap="BuPu")
plt.title('Factor Analysis')
 
factor_variance = fa.get_factor_variance()#贡献率
fa_score = fa.transform(data)#因子得分
 
#综合得分
complex_score=np.zeros([fa_score.shape[0],])
for i in range(n_factors):
    complex_score+=fa_score[:,i]*factor_variance[1][i]#综合得分s

