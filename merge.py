import pandas as pd
import numpy as np
from transformers import pipeline
import statistics as st
# import random

headlines = pd.read_csv("CF-AN-equities-TCS-03-May-2025_modded.csv")
stocks = pd.read_csv("TCS_STOCKS!2.csv")

# print(headlines.head())
date_stocks = list(stocks['Date'])    #Convert the dates into a list
date_headlines = list(headlines['BROADCAST DATE/TIME'])   #convert the headline dates into list
print(date_headlines[50])
print(date_stocks[50])

print(len(stocks['Date']))
# if date_stocks[246] not in date_headlines:
#     stocks.drop(246,inplace=True)
#     print("HI")
# print(stocks['Date'])


rows_to_del = []

for i in range(len(date_stocks)):
    if date_stocks[i] not in date_headlines:  #Check if if there in the data_headlines if not then delete entire row
        rows_to_del.append(i)
stocks.drop(index=rows_to_del,inplace=True)
date_stocks = list(stocks['Date'])


# print(len(stocks['Date']))
#Take a dictionary and the value as a list
#Calculation for sentiment
#Initialization for sentiment
nlp = pipeline("sentiment-analysis",model="ProsusAI/finbert",device = 0)  #Object initialization
sentiment_dict = {}

def map_sentiment(senti):
    label = senti[0]['label'].lower()
    score = senti[0]['score']
    
    if label == "positive":
        return score
    elif label == "negative":
        return -score
    else:
        return 0

for date in date_headlines:
    sentiment_dict[date] = []   #initializing

for i in range(len(date_headlines)):
    sentiment_dict[headlines['BROADCAST DATE/TIME'][i]].append(map_sentiment(nlp(headlines['DETAILS'][i])))

sentiment_stock_making = []
for date in date_stocks:
    sentiment_stock_making.append(st.mean(sentiment_dict[date]))

stocks['Sentiment'] = sentiment_stock_making
print(stocks['Sentiment'])

#Now save the file
stocks.to_csv("Stocks_TCS2_date.csv")