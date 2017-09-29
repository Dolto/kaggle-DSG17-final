import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn import pipeline
from sklearn import preprocessing
from sklearn.externals import joblib
from splitter import SplittingEstimator
import xgboost as xgb


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

# Create a validation set with 20% of the training set
# X_fit, X_val, y_fit, y_val = model_selection.train_test_split(X_train, y_train, test_size=0.2)
X_fit, X_val, y_fit, y_val = custom_train_test_split(X_train, y_train)
print('X_fit:', X_fit.shape)
print('X_val:', X_val.shape)
print('y_fit:', y_fit.shape)
print('y_val:', y_val.shape)

pipe = pipeline.Pipeline([
    ('gbm', xgb.XGBRegressor(
        n_estimators=2000,
        learning_rate=0.01,
        max_depth=8
    ))
])

pipe.fit(
    X_fit,
    y_fit,
    gbm__eval_set=[(X_fit, y_fit), (X_val, y_val)],
    gbm__eval_metric=['mae'],
    gbm__early_stopping_rounds=10,
    gbm__verbose=True
)

joblib.dump(pipe, 'models/xgboost/pipeline.pkl')
