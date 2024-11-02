#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
#%%
df = pd.read_excel("/Users/fsa/Projects/ISTL 5052 Istatiksel Ogrenme Teorisi/KNN/Immunotherapy1.xlsx")
print(df.head())
print(df.shape)
print(df.info())
print(df.describe())

#%%
X = df.iloc[:,:-1].values
y = df.iloc[:,-1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
#%%
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)
y_train = scaler.fit_transform(y_train.reshape(-1,1))
y_test = scaler.fit_transform(y_test.reshape(-1,1))

#%%
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5, metric='minkowski',p=2)
knn.fit(X_train, np.ravel(y_train))
y_pred = knn.predict(X_test)
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, f1_score, precision_score, recall_score
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(accuracy_score(y_test, y_pred))
print(precision_score(y_test, y_pred))
print(recall_score(y_test, y_pred))
print(f1_score(y_test, y_pred))

#%% uygun k değerinni belirlenmesi
hatalar=[]
for i in range(1,31):
    knn1 = KNeighborsClassifier(n_neighbors=i)
    knn1.fit(X_train, np.ravel(y_train))
    predict_i = knn1.predict(X_test)
    hatalar.append(np.mean(predict_i != y_test))

# grafik çizimi
plt.figure(figsize=(15,6))
plt.plot(range(1,31), hatalar,".",color="Blue",linestyle="--")
plt.title("1-30 Aralığında Bulunan k Değerlerine Karşılık Gelen Hata Oranları")
plt.xlabel("k Sayısı")
plt.ylabel("Hata Oranları")
plt.show()

#%%
knn2 = KNeighborsClassifier(n_neighbors=13)
knn2.fit(X_train, np.ravel(y_train))
predict_i2 = knn2.predict(X_test)
print(accuracy_score(y_test, y_pred))
print(f1_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

#%%
f1 = []
for i in range(1, 31):
    knn1 = KNeighborsClassifier(n_neighbors=i)
    knn1.fit(X_train, np.ravel(y_train))
    predict_i = knn1.predict(X_test)
    f1.append(f1_score(y_test, predict_i))

# grafik çizimi
plt.figure(figsize=(15, 6))
plt.plot(range(1, 31), f1, ".", color="Blue", linestyle="--")
plt.title("1-30 Aralığında Bulunan k Değerlerine Karşılık Gelen F1-Score Oranları")
plt.xlabel("k Sayısı")
plt.ylabel("F1-Score Oranları")
plt.show()

#%%
from sklearn.metrics import roc_curve, auc
ypo,dpo,eski_deger = roc_curve(y_test,predict_i2)
auc = auc(ypo,dpo)
print(auc)
plt.figure(figsize=(15,6))
plt.plot(ypo,dpo,".",color="Blue",linestyle="--")
plt.show()

#%% yeni tahmin
_cinsiyet = 1
_yas = 25
_sure = 11
_adet = 4
_tip = 2
_alan = 100
_sertlesmeCapi = 8
girdiler = np.array([_cinsiyet, _yas, _sure, _adet, _tip, _alan, _sertlesmeCapi])
girdiler = scaler.fit_transform(girdiler.reshape(1,-1))
tahmin = knn2.predict(girdiler)
print(tahmin)
print(girdiler)


