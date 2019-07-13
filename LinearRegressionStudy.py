import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression

l3 = ["Symboling","normalized-losses","Brand","fuel type","Aspiration","No Of Doors","Body Style","Drive Wheel","Engine location","Wheel base","Lenght","Width","Height","Curb weight","Engine Type","No of cylinders","Engine size","fuel System","Bore","stroke","Compression ratio","horsepower","Rpm","City mpg","highway mpg","Price"]
dataset3 = pd.read_csv(r"H:\server\spyder\Datasets\imports-85.csv",names = l3,na_values = "?")
x3 = dataset3.iloc[:,0:25].values
y3 = dataset3["Price"].values

Imp = SimpleImputer(strategy = "most_frequent")    # Simple Imputer (use before label encoding)
x3[:,0:25] = Imp.fit_transform(x3[:,0:25])
y3 = Imp.fit_transform(y3.reshape(-1,1))

labelencoder = LabelEncoder()
x3[:,2] = labelencoder.fit_transform(x3[:,2].astype(str))
x3[:,3] = labelencoder.fit_transform(x3[:,3].astype(str))
x3[:,4] = labelencoder.fit_transform(x3[:,4].astype(str))
x3[:,5] = labelencoder.fit_transform(x3[:,5].astype(str))
x3[:,6] = labelencoder.fit_transform(x3[:,6].astype(str))
x3[:,7] = labelencoder.fit_transform(x3[:,7].astype(str))
x3[:,8] = labelencoder.fit_transform(x3[:,8].astype(str))
x3[:,14] = labelencoder.fit_transform(x3[:,14].astype(str))
x3[:,15] = labelencoder.fit_transform(x3[:,15].astype(str))
x3[:,17] = labelencoder.fit_transform(x3[:,17].astype(str))

lin_reg = LinearRegression()
lin_reg.fit(x3,y3)
lin_reg.score(x3,y3)

hp = dataset3["horsepower"]
plt.scatter(hp,y3)
