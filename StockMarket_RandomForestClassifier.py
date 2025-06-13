import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,mean_squared_error,classification_report
from sklearn.model_selection import train_test_split
from sklearn import tree

stocks = pd.read_csv("Stocks_TCS2.csv")

stocks['Target'] = (stocks['Open'].shift(-1) > stocks['Open']).astype(int)
#More signals or we can call them lag features!
stocks['lag_1'] = stocks['Open'].shift(1)
stocks['lag_2'] = stocks['Open'].shift(2)
stocks['rolling_3'] = stocks['Open'].rolling(3).mean()
stocks['per% _change'] = stocks['Close'].pct_change()
#There is a huge impact on accuracy score when the lag features are used! from 55.3 to ~70 is huge ,,, great

stocks = stocks.dropna()

X = stocks.drop(['Target'],axis = 1)
Y = stocks['Target']

x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.2,shuffle=False,random_state=53)

forestClass = RandomForestClassifier(n_estimators=200,random_state=53,class_weight='balanced')
forestClass.fit(x_train,y_train)
y_pred = forestClass.predict(x_test)

#Plotting a decision tree
feature_name = stocks.columns.tolist()
plt.figure(figsize=(80,50))
tree.plot_tree(forestClass.estimators_[0],feature_names=feature_name,filled=True)

print(f"The accuracy score is : {accuracy_score(y_test,y_pred) * 100}")
print(f"Mean squared error : {mean_squared_error(y_test,y_pred)}")
print(f"{classification_report(y_test,y_pred)}")
# print(stocks['Target'].value_counts(normalize=True))
plt.show()