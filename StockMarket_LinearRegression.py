import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error

#Initializing the dataframe
stocks = pd.read_csv("Stocks_TCS2_date.csv")   #This is a dataframe
# print(stocks.head())

stocks['Target'] = stocks['Open'].shift(-1)
#More signals or we can call them lag features!
# stocks['lag_1'] = stocks['Open'].shift(1)
# stocks['lag_2'] = stocks['Open'].shift(2)
# stocks['rolling_3'] = stocks['Open'].rolling(3).mean()
# stocks['per% _change'] = stocks['Close'].pct_change()
#By using the lag features the linear regression's r2_score decreased by 0.0002 and mean squared error increased by 30
#So better dont use this
stocks = stocks.dropna()

X = stocks.drop(['Target','Date'], axis = 1)   #The input features
Y = stocks['Target']  #Training them on the next day's closeing price!
#Split the data
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.2,random_state=32,shuffle=False)

liReg = LinearRegression()
liReg.fit(x_train,y_train)
y_pred = liReg.predict(x_test)
y_train_pred = liReg.predict(x_train)  #predicting the x_trian values

# print("NaNs in y_test:", np.isnan(y_test).sum())
# print("The r2 train score is : ", r2_score(y_train,y_train_pred))
print("The r2 test score is : ", r2_score(y_test,y_pred))
print(f"Mean squared Error  : {mean_squared_error(y_test,y_pred)}")

#Plotting the Regression scatter plot
plt.scatter(y_test,y_pred)
plt.plot([min(y_test),max(y_test)],[min(y_test),max(y_test)])
plt.title("Regression scatter plot")
plt.xlabel("Acutal Opening price")
plt.ylabel("Predicted Opening price")

#PLotting the date vs price
plt.figure(1)
plt.figure(figsize=(50,40))
plt.plot(stocks['Date'].iloc[int(len(stocks)*0.8):],y_test,label = "Actual",color = "blue")
plt.plot(stocks['Date'].iloc[int(len(stocks)*0.8):],y_pred,label = "Predicted",color = "orange",linestyle = "--")
plt.title("Actual vs Predicted")
plt.xlabel("Date")
plt.ylabel("Price")

# plt.legend()
plt.show()