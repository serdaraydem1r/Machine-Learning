#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report,roc_curve
#%%
df = pd.read_excel("/Users/fsa/Projects/ISTL 5052 Istatiksel Ogrenme Teorisi/SVM/data2.xlsx")
print(10*"*","Veri İzleme Monütörü",10*"*")
print(df.head())
print(50*"-")
print(df.info())
print(50*"-")
print(df.describe())
print(50*"-")
print(df.isnull().sum())
print(50*"-")
print(df.shape)

#%%
print(df["SR1"].mean()) # 396.081
print(df["SR1"].median()) # 419.025

print(df["Tpingpong"].mean()) # 27.35
print(df["Tpingpong"].median()) #26.965
