# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 20:45:45 2021

@author: mashituo
"""
## description about dataset
'''
http://archive.ics.uci.edu/ml/datasets/Combined+Cycle+Power+Plant
Data Set Information:

The dataset contains 9568 data points collected from a Combined Cycle Power Plant over 6 years (2006-2011), when the power plant was set to work with full load. Features consist of hourly average ambient variables Temperature (T), Ambient Pressure (AP), Relative Humidity (RH) and Exhaust Vacuum (V) to predict the net hourly electrical energy output (EP) of the plant.
A combined cycle power plant (CCPP) is composed of gas turbines (GT), steam turbines (ST) and heat recovery steam generators. In a CCPP, the electricity is generated by gas and steam turbines, which are combined in one cycle, and is transferred from one turbine to another. While the Vacuum is colected from and has effect on the Steam Turbine, he other three of the ambient variables effect the GT performance.
For comparability with our baseline studies, and to allow 5x2 fold statistical tests be carried out, we provide the data shuffled five times. For each shuffling 2-fold CV is carried out and the resulting 10 measurements are used for statistical testing.
We provide the data both in .ods and in .xlsx formats.


Attribute Information:

Features consist of hourly average ambient variables
- Temperature (T) in the range 1.81°C and 37.11°C,
- Ambient Pressure (AP) in the range 992.89-1033.30 milibar,
- Relative Humidity (RH) in the range 25.56% to 100.16%
- Exhaust Vacuum (V) in teh range 25.36-81.56 cm Hg
- Net hourly electrical energy output (EP) 420.26-495.76 MW
The averages are taken from various sensors located around the plant that record the ambient variables every second. The variables are given without normalization.

'''
#import models
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score
#read data
df=pd.read_excel("ccpp.xlsx")
X=df[['AT','V','AP','RH']]
y=df.PE
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)
#train regressor
lr=LinearRegression()
lr.fit(X_train,y_train)
y_predict=lr.predict(X_test)
#evaluate
x=np.arange(len(y_test))
plt.plot(x,y_predict,'ro')
plt.plot(x,y_test,'bo')
print('the coefficient and bias:',lr.coef_,lr.intercept_)
print('MSE:',mse(y_predict,y_test))
print('R2-score',r2_score(y_test,y_predict))



