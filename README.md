# 📈 Stock Market Prediction Using Machine Learning

This project demonstrates how different Machine Learning algorithms can be applied to stock market data to predict stock price movements or future values.  
It includes multiple ML models (Regression + Classification) implemented in Python, using a sample dataset (`Stocks_TCS2.csv`).

---

## 🚀 Project Overview

The stock market is highly volatile and nonlinear. This project explores how classical machine-learning algorithms perform on such financial time-series data.  
Each script in the repository loads the dataset, preprocesses it, trains a particular ML model, and evaluates its performance.

### ✔️ Goals of the Project
- Predict future stock prices or stock movement direction  
- Compare classical ML algorithms  
- Understand how data preprocessing affects prediction quality  
- Build reusable scripts for stock-market ML experimentation  

---

## 📂 Repository Structure

| File Name | Description |
|----------|-------------|
| **StockMarket_LinearRegression.py** | Linear Regression model for predicting future stock prices |
| **StockMarket_LogisticRegression.py** | Logistic Regression model for predicting UP/DOWN movement |
| **StockMarket_KNN.py** | K-Nearest Neighbors model |
| **StockMarket_KNNReggresor.py** | KNN Regressor for numerical prediction |
| **StockMarket_RandomForestClassifier.py** | Random Forest model for classification |
| **StockMarket_RandomForestRegressor.py** | Random Forest model for regression |
| **StockMarket_SVM.py** | Support Vector Machine model |
| **Stocks_TCS2.csv** | Sample stock dataset (TCS stock history) |
| **README.md** | Project documentation |

---

## 🛠️ Technologies Used

- **Python 3.x**
- **NumPy**
- **Pandas**
- **Scikit-Learn**
- **Matplotlib / Seaborn** *(optional for visualization)*

Install dependencies with:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
