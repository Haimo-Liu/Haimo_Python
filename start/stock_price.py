import pandas as pd
import math
import quandl as ql

pd.set_option('display.max_columns', 20)


url = "/Users/haimo.liu/Documents/python_files/GOOG.csv"
df = pd.read_csv(url)

#df = ql.get("WIKI/GOOGL")

# print(df.head())

df = df[['Open', 'High', 'Low', 'Adj Close', 'Volume']]

df['HL_percent'] = (df['High'] - df['Low']) / df['Low'] * 100
df['change_percent'] = (df['Adj Close'] - df['Open']) / df['Open'] * 100
df = df[['Adj Close', 'HL_percent', 'change_percent', 'Volume']]

df['forecast_col'] = df['Adj Close']
df.fillna(-9999, inplace=True)

forecast_shift = int(math.ceil(0.05*len(df)))

df['label'] = df['forecast_col'].shift(-forecast_shift)

print(df.head())


