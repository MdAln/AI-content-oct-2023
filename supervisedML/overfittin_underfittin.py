# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split, validation_curve
from sklearn.model_selection import learning_curve, ShuffleSplit, cross_validate
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
#%%
df = pd.read_csv('data/dataset_example.csv')
print(df.columns.tolist())
#%%
df.plot(kind='scatter',x='x',y='y',
        figsize=(10,6),
        title='X and Y comparison')
#%% validation
X = df[['x']]
y = df['y']
degrees = [2,3,4,5,6,7,8,9]
model = Pipeline(steps=[
    ('pf',PolynomialFeatures()),
    ('regression',LinearRegression())])
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.3)
train_scores,test_scores = validation_curve(model,X,y, param_name='pf__degree',
                 param_range=degrees,
                 scoring='neg_mean_absolute_error')
#%% plot
score_df = pd.DataFrame({
    'degree': degrees,
    })
score_df [['train1','train2','train3','train4','train5']]=train_scores
score_df [['test1','test2','test3','test4','test5']]=test_scores
score_df.plot(kind='line',x='degree',y=['train1','train2','train3','train4','train5'],style='o-')
#%% average of training and test accuracy
score_df['avg_train_score'] = score_df[['train1','train2','train3','train4','train5']].mean(axis=1)
score_df['avg_test_score'] = score_df[['test1','test2','test3','test4','test5']].mean(axis=1)
ax=score_df.plot(kind='line',x='degree',y='avg_train_score',style='o-',color='green')
score_df.plot(kind='line',x='degree',y='avg_test_score',style='o-',color='red',ax=ax)
#%%
model.set_params(pf__degree=4)
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
print('accuracy',r2_score(y_test, y_pred))
print('mae',mean_absolute_error(y_test, y_pred))
print('mse',mean_squared_error(y_test, y_pred))
y_pred_all = model.predict(X)
df['y2'] = y_pred_all
ax = df.plot.scatter(x='x',y='y')
df.plot.line(x='x',y='y2',color='red',ax=ax,style='x')