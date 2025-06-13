import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score,mean_squared_error
import matplotlib.pyplot as plt

stocks = pd.read_csv("Stocks_TCS2.csv")
#More signals or we can call them lag features!
stocks['lag_1'] = stocks['Open'].shift(1)
stocks['lag_2'] = stocks['Open'].shift(2)
stocks['rolling_3'] = stocks['Open'].rolling(3).mean()
stocks['per% _change'] = stocks['Close'].pct_change()
#By using these lag features i was able to get +0.003 change in r2_score and about -1000 in mean squared error!
#Worth it ? donno..

#Shifting the Opening price
stocks['Target'] = stocks['Open'].shift(-1)
stocks = stocks.dropna()  #Removing the only null value left
# print(stocks.isnull().sum())
X = stocks.drop(['Target'],axis = 1)
Y = stocks['Target']

x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.2,shuffle=False,random_state=53)

#Initializing the Random forest regressor
forest = RandomForestRegressor(n_estimators=100,random_state=53)
#Fitting and predicting the data
forest.fit(x_train,y_train)
y_pred = forest.predict(x_test)

print(f"The r2 score is : {r2_score(y_test,y_pred)}")
print(f"The mean squared error is : {mean_squared_error(y_test,y_pred)}")