import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Imputer
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer

# Adult Data set
l = ["Age","Work Class","Others","Education","Eduction No","Marital Status","Occupation","Relationship Status","Race","Gender","Capital Gain","Capital loss","Hours per week","Native country","Salary"]
dataset = pd.read_csv(r"H:\server\spyder\Datasets\adult.csv",names = l,na_values = " ?")            # na_values function is for change the ? values to nan because we can not work on ? values
x = dataset.iloc[:,0:14].values
labelencoder = LabelEncoder()
x[:,3] = labelencoder.fit_transform(x[:,3].astype(str))    # label encoder convert the data into label for 

# Poker Hands Data set
l2 = ["Suit of card #1","Rank of card #1","Suit of card #2", "Rank of card #2","Suit of card #3","Rank of card #3","Suit of card #4" ,"Rank of card #4","Suit of card #5" , "Rank of card 5","Poker Hand"]
dataset2 = pd.read_csv(r"H:\server\spyder\Datasets\poker-hand-training-true.csv",names = l2 ,na_values = " ?")
x2 = dataset2.iloc[:,0:10].values
Imp = Imputer()
x2[:,0:10] = Imp.fit_transform(x2[:,0:10])
labelencoder = LabelEncoder()
x2[:,3] = labelencoder.fit_transform(x2[:,3].astype(str))


# Imputer on Adult Data set
l = ["Age","Work Class","Others","Education","Eduction No","Marital Status","Occupation","Relationship Status","Race","Gender","Capital Gain","Capital loss","Hours per week","Native country","Salary"]
dataset = pd.read_csv(r"H:\server\spyder\Datasets\adult.csv",names = l,na_values = " ?")
x = dataset.iloc[:,0:14].values
Imp = SimpleImputer(strategy = "most_frequent")    # Simple Imputer (use before label encoding)
x[:,0:14] = Imp.fit_transform(x[:,0:14])
x[27:28,1:2]

labelencoder = LabelEncoder()
x[:,1] = labelencoder.fit_transform(x[:,1].astype(str))
x[:,3] = labelencoder.fit_transform(x[:,3].astype(str))
x[:,5] = labelencoder.fit_transform(x[:,5].astype(str))
x[:,6] = labelencoder.fit_transform(x[:,6].astype(str))
x[:,7] = labelencoder.fit_transform(x[:,7].astype(str))
x[:,8] = labelencoder.fit_transform(x[:,8].astype(str))
x[:,9] = labelencoder.fit_transform(x[:,9].astype(str))
x[:,13] = labelencoder.fit_transform(x[:,13].astype(str))
Imp = Imputer(strategy = "median")
x[:,0:14] = Imp.fit_transform(x[:,0:14])
idx = dataset.index
y = dataset["Age"]
plt.scatter(idx[0:1000],y[0:1000])
x[27:28,1:2]


