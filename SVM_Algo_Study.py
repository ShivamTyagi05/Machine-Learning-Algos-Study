from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

ir = load_iris()
d = ir.data
y = ir.target

X_train,X_test,Y_train,Y_test = train_test_split(d,y)

svc = SVC(kernel = 'poly', degree = 4)
svc.fit(X_train,Y_train)
svc.score(X_train,Y_train)
svc.score(X_test,Y_test)