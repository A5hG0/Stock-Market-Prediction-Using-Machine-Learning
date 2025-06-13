import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,mean_absolute_error,confusion_matrix,ConfusionMatrixDisplay,classification_report,mean_squared_error,RocCurveDisplay
from sklearn.model_selection import train_test_split
# import seaborn as sea

#Reading data
stocks = pd.read_csv("Stocks_TCS2.csv")

#More signals or we can call them lag features!
stocks['lag_1'] = stocks['Open'].shift(1)
stocks['lag_2'] = stocks['Open'].shift(2)
stocks['rolling_3'] = stocks['Open'].rolling(3).mean()
stocks['per% _change'] = stocks['Close'].pct_change()

#In logistic regression adding the lag features didnt make any difference!

stocks['Target'] = (stocks['Open'].shift(-1) > stocks['Open']).astype(int)
# print(stocks['Target'].value_counts())

stocks = stocks.dropna()

X = stocks.drop(['Target'],axis = 1)
Y = stocks['Target']

x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.2,shuffle=False,random_state=54)

#Initializing logistic regression
log_reg = LogisticRegression(max_iter=1200,random_state=54,class_weight='balanced')
log_reg.fit(x_train,y_train)
y_pred = log_reg.predict(x_test)

#Confusion Matrix
cm = confusion_matrix(y_test,y_pred)
dis = ConfusionMatrixDisplay(confusion_matrix = cm,display_labels = ['Down','Up'])
dis.plot(cmap = 'Reds')
plt.title("Confusion Matrix")

#ROC curve display!
RocCurveDisplay.from_estimator(log_reg,X,Y)

#Classification report
print(classification_report(y_test,y_pred,target_names=['Down','Up']))
print(f"Accuracy score : {accuracy_score(y_test,y_pred) * 100}%")
# print(f"Mean Absolute Error : {mean_absolute_error(y_test,y_pred)}")
print(f"Mean Squared Error : {mean_squared_error(y_test,y_pred)}")
plt.show()