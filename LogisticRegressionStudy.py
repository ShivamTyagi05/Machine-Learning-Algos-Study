import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

# Adult dataset
l = ["Age","Work Class","Others","Education","Eduction No","Marital Status","Occupation","Relationship Status","Race","Gender","Capital Gain","Capital loss","Hours per week","Native country","Salary"]
dataset = pd.read_csv(r"H:\server\spyder\Datasets\adult.csv",names = l,na_values = " ?")
x = dataset.iloc[:,0:14].values
y = dataset["Salary"].values

Imp = SimpleImputer(strategy = "most_frequent")    # Simple Imputer (use before label encoding)
x[:,0:14] = Imp.fit_transform(x[:,0:14])

labelencoder = LabelEncoder()
x[:,1] = labelencoder.fit_transform(x[:,1].astype(str))
x[:,3] = labelencoder.fit_transform(x[:,3].astype(str))
x[:,5] = labelencoder.fit_transform(x[:,5].astype(str))
x[:,6] = labelencoder.fit_transform(x[:,6].astype(str))
x[:,7] = labelencoder.fit_transform(x[:,7].astype(str))
x[:,8] = labelencoder.fit_transform(x[:,8].astype(str))
x[:,9] = labelencoder.fit_transform(x[:,9].astype(str))
x[:,13] = labelencoder.fit_transform(x[:,13].astype(str))
y = labelencoder.fit_transform(y.astype(str))

log_reg = LogisticRegression()
log_reg.fit(x,y)
log_reg.predict(x[0,:14].reshape(1,-1))
log_reg.score(x,y)

y_pred = log_reg.predict(x)
cm = confusion_matrix(y,y_pred)

precision = cm[0][0] / (cm[0][0] + cm[1][0])
recall = cm[0][0] / (cm[0][0] + cm[0][1])
score = (2 * precision * recall) / (precision + recall)

edu = np.array([dataset["Eduction No"]])
y2 = np.exp(-edu)
y2 = 1 + y2
y2 = 1 / y2
plt.plot(y2)