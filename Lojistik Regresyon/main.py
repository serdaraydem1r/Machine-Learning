#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('/Users/fsa/Projects/ISTL 5052 Istatiksel Ogrenme Teorisi/Lojistik Regresyon/diabetes.csv')
print(df.head())
print(df.describe())
print(df.info())
print(df.isnull().sum())
print(df["Outcome"].value_counts())
#%%
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report,f1_score,roc_curve,roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import  MinMaxScaler

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
print(X.shape)
print(y.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
print(X_train.shape)
print(y_train.shape)
#%%
"""
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)
y_train = scaler.fit_transform(y_train.reshape(-1,1))
y_test = scaler.fit_transform(y_test.reshape(-1,1))
"""


#%%
lr = LogisticRegression(solver="liblinear").fit(X_train, y_train)
y_pred = lr.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(roc_auc_score(y_test, y_pred))
print(roc_curve(y_test, y_pred))
print(accuracy_score(y_test, y_pred))
print("*"*20)
print(lr.intercept_)
print(lr.coef_)

#%%
print(lr.predict(X)[:-1])
print(y[:-1])

