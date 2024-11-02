#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy import stats
from sklearn.metrics import explained_variance_score, mean_absolute_error,mean_squared_error,r2_score

#%%
df = pd.read_excel("/Users/fsa/Projects/ISTL 5052 Istatiksel Ogrenme Teorisi/Coklu Dogrusal Regresyon/Pv.xlsx")
print(df.head())
print(df.describe())
#%%
X = df.iloc[:,:-1].values
y = df.iloc[:,-1].values
X_train, X_test, y_train,y_test = train_test_split(X, y, test_size=0.3, random_state=0)
print(X_train.shape, y_train.shape)

print(stats.describe(X_train,axis=0))
print(np.std(X_train,axis=0))
print(stats.describe(X_test,axis=0))
print(np.std(X_test,axis=0))

print(stats.describe(y_train,axis=0))
print(np.std(y_train,axis=0))
print(stats.describe(y_test,axis=0))
print(np.std(y_test,axis=0))

#%% Ölçeklendirnme yap
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)
y_test = scaler.fit_transform(y_test.reshape(-1,1))
y_train = scaler.fit_transform(y_train.reshape(-1,1))

#%% Çoklu Doğrusal Uydurma
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train,y_train)
y_pred = lr.predict(X_test)
print(y_pred)
#%% model kat sayıları
print(lr.intercept_)
print(lr.coef_)

#%%
import statsmodels.api as sm
X = np.append(arr=np.ones((1620,1)).astype(int),values=X,axis=1)
X_new = X
lr_ols = sm.OLS(endog=y,exog=X_new).fit()
print(lr_ols.summary()) # x4 çıkarılıcak

#%%
X_opt = X[:,[0,1,2,3,5,6,7]]
new_lr_ols = sm.OLS(endog=y,exog=X_opt).fit()
print(new_lr_ols.summary()) # x3 çıkarılıcak
#%%
X_opt1 = X[:,[0,1,2,5,6,7]]
new2_lr_ols = sm.OLS(endog=y,exog=X_opt1).fit()
print(new2_lr_ols.summary()) # tüm veriler 0.05 değerinden ufak oldu anlam düzeyi.

#%% ileriye doğru seçme 
