import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# Adult dataset
l = ["Age","Work Class","Others","Education","Eduction No","Marital Status","Occupation","Relationship Status","Race","Gender","Capital Gain","Capital loss","Hours per week","Native country","Salary"]
dataset = pd.read_csv(r"H:\server\spyder\Datasets\adult.csv",names = l,na_values = " ?")
x = dataset.iloc[:,0:14].values
y = dataset["Salary"].values

Imp = SimpleImputer(strategy = "most_frequent")    # Simple Imputer (use before label encoding)
x[:,0:14] = Imp.fit_transform(x[:,0:14])

labelencoder_x1 = LabelEncoder()
x[:,1] = labelencoder_x1.fit_transform(x[:,1].astype(str))
labelencoder_x1.inverse_transform(np.array([0,1,2,3,4,5,6,7]))

labelencoder_x2 = LabelEncoder()
x[:,3] = labelencoder_x2.fit_transform(x[:,3].astype(str))
labelencoder_x2.inverse_transform(np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]))

labelencoder_x3 = LabelEncoder()
x[:,5] = labelencoder_x3.fit_transform(x[:,5].astype(str))
labelencoder_x3.inverse_transform(np.array([0,1,2,3,4,5,6]))

labelencoder_x4 = LabelEncoder()
x[:,6] = labelencoder_x4.fit_transform(x[:,6].astype(str))
labelencoder_x4.inverse_transform(np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13]))

labelencoder_x5 = LabelEncoder()
x[:,7] = labelencoder_x5.fit_transform(x[:,7].astype(str))
labelencoder_x5.inverse_transform(np.array([0,1,2,3,4,5]))

labelencoder_x6 = LabelEncoder()
x[:,8] = labelencoder_x6.fit_transform(x[:,8].astype(str))
labelencoder_x6.inverse_transform(np.array([0,1,2,3,4]))

labelencoder_x7 = LabelEncoder()
x[:,9] = labelencoder_x7.fit_transform(x[:,9].astype(str))
labelencoder_x7.inverse_transform(np.array([0,1]))

l = []
labelencoder_x8 = LabelEncoder()
x[:,13] = labelencoder_x8.fit_transform(x[:,13].astype(str))
for i in x[:,13]:
    if i not in l:
        l.append(i)
    
labelencoder_x8.inverse_transform(np.array(l))

labelencoder = LabelEncoder()
y = labelencoder.fit_transform(y.astype(str))
labelencoder.inverse_transform(np.array([0,1]))

X_train,X_test,Y_train,Y_test = train_test_split(x,y,random_state=1)    

Kn = KNeighborsClassifier(n_neighbors = 10)
Kn.fit(X_train,Y_train)
Kn.score(X_train,Y_train)
Kn.score(X_test,Y_test)
