#%%
import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor # çoklu bağlantıya bakmaya yarar.
from patsy import dmatrices # regresyon modelini girmeye yarar
#%%
df = pd.read_excel("/Users/fsa/Projects/ISTL 5052 Istatiksel Ogrenme Teorisi/Coklu Dogrusal Regresyon/VeriOnIsleme_2.xlsx")
print(df.head())
print(df.tail())
print(df.describe())
print(df.info())
#%%  Çoklu bağlantı var mı ? Ona bakıcaz ilk.
y,X = dmatrices('ESS~BEKY+SGE+SGY+BA', data = df, return_type='dataframe')
vif = pd.DataFrame()
vif["VIF Değeri"] = [variance_inflation_factor(X.values,i ) for i in range(X.shape[1])]
vif["Özellikleri"] = X.columns
print(vif.head())
"""
   VIF Değeri Özellikleri
0   73.633129   Intercept
1    9.432633        BEKY
2   10.072963         SGE
3    6.364627         SGY
4   11.417390          BA
çoklu regresyona uygun değil.

varsayımlar : 

otokorelasyon hataların arasında bağ var mı ? darwin-watson yöntemi
hataların normal dağılıyor mu ? kolmogorv shapirnov, 
eş varyans sorunu var mı ? eğri etrafındaki dağılıma bakarız.

backward elımınaton bak.
"""

#%%


