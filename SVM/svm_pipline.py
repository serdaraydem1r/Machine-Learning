#%%
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import LinearSVC
import numpy as np
#%%
X,y = make_moons(n_samples=100, noise=.15, random_state=42)
poly_svm_class = make_pipeline(PolynomialFeatures(3), StandardScaler(),LinearSVC(C=50,max_iter=10_000,random_state=42))
poly_svm_class.fit(X,y)

def plot_dataset(X,y,axes):
    plt.plot(X[:,0][y==0],X[:,1][y==0],"bs")
    plt.plot(X[:,0][y==1],X[:,1][y==1],"g^")
    plt.axis(axes)
    plt.grid(True)
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")

def plot_predition(clf,axes):
    x0s=np.linspace(axes[0],axes[1],100)
    x1s=np.linspace(axes[2],axes[3],100)
    x0,x1 = np.meshgrid(x0s,x1s)
    X=np.c_[x0.ravel(),x1.ravel()]
    y_pred = clf.predict(X).reshape(x0.shape)
    y_decision = clf.decision_function(X).reshape(x0.shape)
    plt.contour(x0,x1,y_pred,cmap=plt.cm.brg,alpha=0.2)
    plt.contour(x0,x1,y_decision,cmap=plt.cm.brg,alpha=0.1)
plot_predition(poly_svm_class,[-1.5,2.5,-1,1.5])
plot_dataset(X,y,axes=[0-1.5,2.5,-1,1.5])
plt.show()

#%%
print(f"x boyutu: {len(X)}")
print(f"y boyutu: {len(y)}")

#%%
X_yeni = [[1.5,-.5]]
plt.plot(1.5,-.5,"ro")
tahmin = poly_svm_class.predict(X_yeni)
print("Yeni X değeri için tahmin : ",tahmin)