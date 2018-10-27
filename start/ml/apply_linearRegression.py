import pandas as pd
import math
import quandl as ql
import numpy as np
import datetime
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')

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

forecast_shift = int(math.ceil(0.01*len(df)))

df['label'] = df['forecast_col'].shift(-forecast_shift)

#df.dropna(axis=0, inplace=True)

#print(df.head())
print(df.tail())

x = np.array(df.drop(['label'], axis=1))
x_lately = x[-forecast_shift:]

x = x[:-forecast_shift]

y = np.array(df['label'])
y = y[:-forecast_shift]



#print(forecast_shift)
#print(len(x), len(y))

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

clf = LinearRegression(n_jobs=10)
#clf = svm.SVR(kernel='poly')
clf.fit(x_train, y_train)
accuracy = clf.score(x_test, y_test)

#print(accuracy)

forecast_set = clf.predict(x_lately)

print(forecast_set, accuracy, forecast_shift)

plt.plot(forecast_set)
plt.show()

