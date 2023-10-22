# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.metrics import mean_squared_error
#%%
url = 'https://raw.githubusercontent.com/digipodium/Datasets/main/regression/dataA.csv'
df = pd.read_csv(url)
#%%
X = df[['x']] # X should always be => 2d data
y = df['y']
X_train, X_test, y_train, y_test= train_test_split(X,y,test_size=.2,random_state=0)
#%%
model1 = LinearRegression()
#%%
Xp =  PolynomialFeatures(degree=2).fit_transform(X)
model1.fit(X_train,y_train)
y_pred = model1.predict(X_test)
score = r2_score(y_test, y_pred)*100
mse1= mean_squared_error(y_test, y_pred)
mae1 = mean_absolute_error(y_test, y_pred)