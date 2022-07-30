# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 09:33:52 2021

@author: mashituo
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,accuracy_score,roc_curve,auc,precision_recall_curve
df=pd.read_csv("breast-cancer-wisconsin.csv")
df=df.replace('?',np.nan)
df=df.dropna()
X=df[['age','menopause','size','inv','nodes','caps','malig','breast','quad']]
y=df.irradit
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
#train regressor
lr=LogisticRegression()
lr.fit(X_train,y_train)
y_predict=lr.predict(X_test)
#evaluate
print(confusion_matrix(y_test,y_predict))
print(accuracy_score(y_test,y_predict))
FPR,TPR,threshold=roc_curve(y_test,y_predict,pos_label=4)
#data[:,1]测试集的结果,data[:,2]模型预测的结果,pos_label=1表示正样本
#AUC值计算
AUC=auc(FPR,TPR)
#ROC曲线绘制
plt.figure()
plt.title('ROC CURVE (AUC={:.2f})'.format(AUC))
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.0])
plt.plot(FPR,TPR,color='g')
plt.show()
P,R,threshold=precision_recall_curve(y_test, y_predict,pos_label=4)
plt.plot(P,R,'r--')
plt.show()