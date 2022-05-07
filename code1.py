import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import joblib
from sklearn.metrics import r2_score
dataset=pd.read_csv('datasetcanon.csv')
plt.plot(dataset['Date'],dataset['Close'],'r')
plt.xlabel('Date')
plt.ylabel('Close')
plt.title("Date vs close")
plt.savefig('variations.jpg')
dataset=dataset.iloc[:,1:]
dataset=dataset.drop('Adj Close',axis=1)
z=np.abs(stats.zscore(dataset))
Q1=dataset.quantile(0.25)
Q3=dataset.quantile(0.75)
IQR=Q3-Q1
dataset2=dataset[~((dataset<(Q1-1.5*IQR))|(dataset>(Q3+1.5*IQR))).any(axis=1)]
y=dataset2.iloc[:,3]
x=dataset2.drop('Close',axis=1)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
ml=LinearRegression()
ml.fit(x_train,y_train)
y_pred=ml.predict(x_test)
l=r2_score(y_test,y_pred)
dt=DecisionTreeRegressor(criterion="mse",random_state=0)
dt.fit(x_train,y_train)
Y_pred=dt.predict(x_test)
d=r2_score(y_test,y_pred)
rf=RandomForestRegressor(n_estimators=100,criterion="mae",random_state=0)
rf.fit(x_train,y_train)
y_pred=rf.predict(x_test)
r=r2_score(y_test,y_pred)
if l>r and l>d:
    joblib.dump(ml,'model.save')
elif d>r:
    joblib.dump(dt,'model.save')
else:
    joblib.dump(rf,'model.save')

