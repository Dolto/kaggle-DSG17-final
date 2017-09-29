import math

import fbprophet
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import model_selection


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


data = pd.read_csv('airline-passengers.csv')

data.rename(columns={'passengers': 'y', 'month': 'ds'}, inplace=True)
# data['prophet_yhat'] = 0

# t_split = model_selection.TimeSeriesSplit(max_train_size=12)

# for i in range(12, len(data)):
#     model = fbprophet.Prophet(weekly_seasonality=False)
#     model.add_seasonality(name='monthly', period=30.5, fourier_order=1)
#     model.fit(data[i - 24:i])
#     data['prophet_yhat'][i] = model.predict(data.iloc[[i]])['yhat']

train_test_split_index = 12
train = data[:-train_test_split_index].copy()
test = data[-train_test_split_index:].copy()

model = fbprophet.Prophet(weekly_seasonality=False)
model.add_seasonality(name='monthly', period=30.5, fourier_order=1)
model.fit(train)

pred = model.predict(test)

print('MAPE: {:.3f}'.format(mean_absolute_percentage_error(test['y'], pred['yhat'])))




train_test_split_index = 12

import datetime as dt

series = pd.Series(
    data=data['y'].tolist(),
    index=pd.DatetimeIndex([dt.datetime.strptime(m, '%Y-%m') for m in data['ds']]),
    dtype=np.float
)
import xam

alpha = 0.1
beta = 0.2
gamma = 0.6

train = series[:-train_test_split_index].copy()
test = series[-train_test_split_index:].copy()

pred = xam.tsa.TripleExponentialSmoothingForecaster(
    alpha,
    beta,
    gamma,
    season_length=12,
    multiplicative=True
).fit(train).predict(test.index)

print('MAPE: {:.3f}'.format(mean_absolute_percentage_error(test, pred)))
