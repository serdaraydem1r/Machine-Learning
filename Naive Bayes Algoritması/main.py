#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report

df = pd.read_excel("/Users/fsa/Projects/ISTL 5052 Istatiksel Ogrenme Teorisi/KNN/Immunotherapy1.xlsx")
print(df.head())
print("*"*50)
print(df.info())
print("*"*50)
print(df.describe())

#%%
X = df.iloc[:,:-1].values
y = df.iloc[:,-1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)
y_test = scaler.fit_transform(y_test.reshape(-1,1))
y_train = scaler.fit_transform(y_train.reshape(-1,1))

#%%
gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)