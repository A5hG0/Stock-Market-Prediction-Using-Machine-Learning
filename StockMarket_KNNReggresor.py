import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.preprocessing import StandardScaler

stocks = pd.read_csv("Stocks_TCS2.csv")

stocks['Target'] = stocks['Open'].shift(-1)

#Lag features in action
stocks['lag_1'] = stocks['Open'].shift(1)
stocks['lag_2'] = stocks['Open'].shift(2)
stocks['rolling_3'] = stocks['Open'].rolling(3).mean()
stocks['per%_change'] = stocks['Close'].pct_change()
#By using the lag features we were able to increase the r2_Score by ~0.01 and mean squared error by ~800
stocks = stocks.dropna()

X = stocks.drop(['Target'],axis = 1)
Y = stocks['Target']

x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.2,shuffle=False,random_state=53)

#Initializing the standard scaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#Initializing the model
knn = KNeighborsRegressor(n_neighbors=10)
knn.fit(x_train,y_train)
y_pred = knn.predict(x_test)

print(f"r2_score is : {r2_score(y_test,y_pred)}")
print(f"mean squared error is : {mean_squared_error(y_test,y_pred)}")