from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

ir = load_iris()
d = ir.data

ss = StandardScaler()
d = ss.fit_transform(d)

print(d)

pca = PCA(n_components = 2)
principleComponents = pca.fit_transform(d)
principleDF = pd.DataFrame(principleComponents,columns = ["principleColumn1","principleColumn2"])

y = ir.target
X_train,X_test,Y_train,Y_test = train_test_split(principleDF,y)

df = DecisionTreeClassifier(max_depth = 3)       # max_depth is used to reduce ovwrfitting
df.fit(X_train,Y_train)
df.score(X_train,Y_train)
df.score(X_test,Y_test)

kn = KNeighborsClassifier(n_neighbors = 10)
kn.fit(X_train,Y_train)
kn.score(X_train,Y_train)
kn.score(X_test,Y_test)