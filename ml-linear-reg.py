import time
import datetime
import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd
import quandl, math
import numpy as np
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression

style.use('ggplot')

df = quandl.get("WIKI/AAPL")
df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]
df['PCT_HL_DIFF'] = (df['Adj. Close'] - df['Adj. Low'])/ df['Adj. Close'] * 100.0
df['PCT_DELTA'] = (df['Adj. Close'] - df['Adj. Open'])/ df['Adj. Open'] * 100.0

# Make new df with 2 derived features (PCT_HL_DIFF & PCT_DELTA)
df = df[['Adj. Close', 'PCT_HL_DIFF', 'PCT_DELTA', 'Adj. Volume']]
forecast_col = 'Adj. Close'
df.fillna(value=-99999, inplace=True)
forecast_out = int(math.ceil(0.01 * len(df)))

# Construct X, the features. Preprocess features (scale to between -1 and 1)
X = np.array(df)
X = preprocessing.scale(X)
X_lately = X[-forecast_out:]
X = X[:-forecast_out]

# Construct y, the labels used in supervised learning ML model. every y is forecast_out number of days ahead of its correspoding feature in X
df['Label'] = df[forecast_col].shift(-forecast_out)
df.dropna(inplace=True)
y = np.array(df['Label'])

# Data split, and train. test data not used but can be used to generate confidence
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)
clf = LinearRegression(n_jobs=-1)
clf.fit(X_train, y_train)

# Predict using trained model
forecast_set = clf.predict(X_lately)
df['Forecast'] = np.nan

# Prepare dates for nice representation in df
last_date = df.iloc[-1].name
last_unix = time.mktime(last_date.to_pydatetime().timetuple())
one_day = 86400
next_unix = last_unix + one_day

# Fill in df with predicted values
for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)]+[i]

df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()
