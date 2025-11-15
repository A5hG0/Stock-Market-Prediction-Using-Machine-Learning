# from textblob import TextBlob as TB
# text = TB("TCS share price hits 52-week low; brokerage cuts target- Top Reasons")
# print(text.sentiment)



























import yfinance as yf
import pandas as pd

tcs = yf.download("TCS.NS", start="2015-04-30", end="2025-04-30")  #This is a dataframe
# print(type(tcs))
# # df = pd.DataFrame(tcs.data,columns=tcs.feature_names)
# print(tcs.tail())
tcs.to_csv("TCS_STOCKS!2.csv")

# from transformers import pipeline as pl

# nlp = pl("sentiment-analysis",model="ProsusAI/finbert",device = 0)
# res = nlp("Tata Consultancy Services Limited has informed the Exchange regarding 'Press Release - Jazeera Airways Partners with Tata Consultancy Services to Power AI-led Transformation, Reimagine Digital Customer Experience'.")
# res1 = nlp("Press Release - Jazeera Airways Partners with Tata Consultancy Services to Power AI-led Transformation, Reimagine Digital Customer Experience")
# print(res)
# print(res1)