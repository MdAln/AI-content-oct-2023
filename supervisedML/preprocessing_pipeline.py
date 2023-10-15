#%%
import pandas as pd
import numpy as np

url = 'https://raw.githubusercontent.com/digipodium/Datasets/main/regression/automobile.csv'
df = pd.read_csv(url)

#%%
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

#%%
x= df.drop(columns=['price'])
y = df['price']
#%%
x.replace('?',np.nan,inplace=True)
#%%
wierd_cols = ['normalized-losses','bore','stroke','horsepower','peak-rpm']
for col in wierd_cols:
    x[col]=x[col].astype(float)

#%%
num_cols = x.select_dtypes(include=np.number).columns
cat_cols = x.select_dtypes(exclude=np.number).columns

#%%
cat_ord_col = []
cat_hot_col =[]
for col in cat_cols:
    print(col,x[col].nunique())
    if x[col].nunique()>2:
        cat_hot_col.append(col)
    else:
        cat_ord_col.append(col)
        
#%%
num_pipe = Pipeline(steps=[
    
    ('impute',SimpleImputer()),
    ('scale',StandardScaler()),
    ])
cat_ord_pipe = Pipeline(steps=[
    ('impute',SimpleImputer(strategy='most_frequent')),
    ('scale',OrdinalEncoder()),
    ])

cat_hot_pipe = Pipeline(steps=[
    ('impute',SimpleImputer(strategy='most_frequent')),
    ('encode',OneHotEncoder(drop='first')),
    ])

transformer = ColumnTransformer(
    transformers=[
        ('numerical',num_pipe,num_cols),
        ('categorical_hot', cat_hot_pipe,cat_hot_col),
        ('categorical_ord', cat_ord_pipe,cat_ord_col)]
    )
#%%
xpro = transformer.fit_transform(x)