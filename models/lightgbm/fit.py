import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn import pipeline
from sklearn import preprocessing
from sklearn.externals import joblib
import lightgbm as lgb


# Load data
X_train = pd.read_csv('data/X_train.csv', sep=';')
y_train = pd.read_csv('data/y_train.csv', sep=';')['OrderQty']


def custom_train_test_split(X_train, y_train):
    idx_val = (X_train.year == 2017)
    X_val = X_train[idx_val]
    y_val = y_train[idx_val]
    X_fit = X_train[~idx_val]
    y_fit = y_train[~idx_val]
    return X_fit, X_val, y_fit, y_val

X_fit, X_val, y_fit, y_val = custom_train_test_split(X_train, y_train)

pipe = pipeline.Pipeline([
    ('gbm', lgb.LGBMRegressor(
        objective='regression',
        learning_rate=0.01,
        n_estimators=5000,
        boosting='dart'
    ))
])

pipe.fit(
    X_fit,
    y_fit,
    gbm__eval_set=[(X_fit, y_fit), (X_val, y_val)],
    gbm__eval_metric=['l1'],
    gbm__early_stopping_rounds=30,
    gbm__verbose=True
)

joblib.dump(pipe, 'models/lightgbm/pipeline.pkl')
