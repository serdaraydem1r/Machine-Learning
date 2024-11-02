import pandas as pd
import warnings
import numpy as np
warnings.filterwarnings("ignore",category=UserWarning,module='openpyxl')
df = pd.read_excel('/Users/fsa/Projects/ISTL 5052 Istatiksel Ogrenme Teorisi/Veri OnIsleme/VeriOnIsleme.xlsx')
X = df.iloc[:,:-1].values
y = df.iloc[:,-1].values

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer = imputer.fit(X[:,1:5])
X[:,1:5]=imputer.transform[X[:,1:5]]