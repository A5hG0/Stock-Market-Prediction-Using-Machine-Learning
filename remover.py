import pandas as pd
stocks = pd.read_csv("Stocks_TCS3.csv")

row_to_rem = []
for i in range(len(stocks['Close'])):
    if stocks['Sentiment'][i] == 0:
        row_to_rem.append(i)

#Removing the 0's
stocks.drop(index=row_to_rem,inplace=True)
stocks.to_csv("Stocks_TCS4.csv")