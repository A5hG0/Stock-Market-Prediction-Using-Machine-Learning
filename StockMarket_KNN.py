import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier  #There is also kneighbour regressor
from sklearn.metrics import accuracy_score,classification_report
from sklearn.preprocessing import StandardScaler

stocks = pd.read_csv("Stocks_TCS2.csv")
#The lag features for model training
stocks['lag_1'] = stocks['Open'].shift(1)
stocks['lag_2'] = stocks['Open'].shift(2)
stocks['rolling_3'] = stocks['Open'].rolling(3).mean()
stocks['per%_change'] = stocks['Close'].pct_change()
stocks['Target'] = (stocks['Open'].shift(-1) > stocks['Open']).astype(int)
#The lag features here also very important as when included the accuracy score went up from 50% to 65% which is very much significant

stocks = stocks.dropna()

X = stocks.drop(['Target'],axis = 1)
Y = stocks['Target']

x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.2,shuffle=False,random_state=53)

#standardscaler for knn initialization
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#initialization of knn
knn = KNeighborsClassifier(n_neighbors=10)  #at n neigbours = 10 getting around 70% accuracy
knn.fit(x_train,y_train)

y_pred = knn.predict(x_test)

#printing the results
print(f"Accuracy score is : {accuracy_score(y_test,y_pred)}")
print(classification_report(y_test,y_pred))