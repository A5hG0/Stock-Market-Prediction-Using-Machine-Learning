import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,mean_squared_error,classification_report
from sklearn import tree

stocks = pd.read_csv("Stocks_TCS2.csv")

#Lag features !
stocks['lag_1'] = stocks['Open'].shift(1)
stocks['lag_2'] = stocks['Open'].shift(2)
stocks['Rolling_3'] = stocks['Open'].rolling(3).mean()
stocks['per%_change'] = stocks['Close'].pct_change()

stocks['Target'] = (stocks['Open'].shift(-1) > stocks['Open']).astype(int)  #Initializing the target column
stocks = stocks.dropna()

X = stocks.drop(['Target'],axis=1)  #Taking everything except the target column!
Y = stocks['Target']
# print(stocks.isnull().sum())
#Split the data
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.2,shuffle=False,random_state=53)

svm = SVC(kernel='rbf',C=100,gamma=0.1)
svm.fit(x_train,y_train)
y_pred = svm.predict(x_test)

#plotting the svm
# plt.figure(figsize=(30,20))
# plt.plot(y_test.values,label = 'Actual',marker = 'o')
# plt.plot(y_pred,label = 'Predicted',marker = 'x')
# plt.title("SVM STOCK UP/DOWN PREDICTION")
# plt.xlabel("Sample Index")
# plt.ylabel("Direction up/down")
# plt.legend()
# plt.show()

print(f"The Accuracy is : {accuracy_score(y_test,y_pred)}")
print(f"Mean squared error is : {mean_squared_error(y_test,y_pred)}")
print(f"{classification_report(y_test,y_pred)}")

#This is worse than deciding on coin toss!  ,,,, XD