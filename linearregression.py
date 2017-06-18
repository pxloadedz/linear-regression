#model data using a straight line y = mx+b
#find out what m and b is
#continuous data, supervised learning -> features (attributes) and label (prediction)
#get rid of useless features

import pandas as pd
import quandl, math, datetime
import numpy as np
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
import _pickle as pickle
#preprocessing gets features to have value -1 to 1 for accuracy and processing speed
#can use svm to do regression; also easy to change algorithm

style.use('ggplot')

df = quandl.get('WIKI/GOOGL')

df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]
#in this case, relationship between features (e.g. high-low => volatility, open-close => price up/down?)
#is more important than the feature itself
#in linear regression the input is a feature as it is, so we'll have to compute the r/s as features

#define new column
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Low']) / df['Adj. Low'] * 100.0
df['PCT_Change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0

df = df[['Adj. Close', 'HL_PCT', 'PCT_Change', 'Adj. Volume']]

#label -> price at some point in the future
forecast_col = 'Adj. Close'
df.fillna(-99999, inplace=True) #we can't work with NaN data; doing this will treat na data as outlier

forecast_out = int(math.ceil(0.01*len(df)))

#for each row, we want to predict the future price e.g. 10 days from that day
df['label'] = df[forecast_col].shift(-forecast_out)

x = np.array(df.drop(['label'], 1)) #features are everything except label col
x = preprocessing.scale(x) #normalize dataset along any axis; could potentially add to processing time
x_lately = x[-forecast_out:]
x = x[:-forecast_out]

#df.dropna(inplace=True) #dropped entire row where there's NaN value
y = np.array(df['label'])
y = y[~np.isnan(y)]

x_train, x_test, y_train, y_test = cross_validation.train_test_split(x, y, test_size=0.2)

clf = LinearRegression(n_jobs=-1)
# n_jobs has to do with threading, how many job is running in parallel, default is 1
# clf = svm.SVR(kernel='linear') #trying a different classifier; seems like it needs preprocessing first
clf.fit(x_train, y_train) #fit = train

# save classifier at this point to save training time
# can train classifier maybe once a month
with open('linearregression.pickle', 'wb') as f:
	pickle.dump(clf, f)

#to use the classifier
pickle_in = open('linearregression.pickle', 'rb')
clf = pickle.load(pickle_in)

accuracy = clf.score(x_test, y_test) #score = test
forecast_set = clf.predict(x_lately)
#print(forecast_set, accuracy, forecast_out)
df['Forecast'] = np.nan

last_date = df.iloc[-1].name #2017-04-24 00:00:00
last_unix = last_date.timestamp() #1492966800.0
one_day = 86400 #number of secs in one day
next_unix = last_unix + one_day

for i in forecast_set:
	next_date = datetime.datetime.fromtimestamp(next_unix)
	next_unix += one_day
	df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]

df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()
