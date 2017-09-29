import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn import pipeline
from sklearn import preprocessing
from sklearn.base import clone
from sklearn.externals import joblib
import xgboost as xgb




# Load data
X_train = pd.read_csv('data/X_train.csv', sep=';')
y_train = pd.read_csv('data/y_train.csv', sep=';')

X_train['Material'] = y_train['Material'].copy()
X_train['SalOrg'] = y_train['SalOrg'].copy()


pipe = pipeline.Pipeline([
    ('gbm', xgb.XGBRegressor(
        n_estimators=2000,
        learning_rate=0.05,
        max_depth=13,
        subsample=0.8,
        tree_method='exact'
    ))
])

X_train['SPLIT'] = X_train['Material']
split_keys = np.unique(X_train['SPLIT'])
estimators = {key: clone(pipe) for key in split_keys}

for i, key in enumerate(split_keys):

    mask = X_train['SPLIT'] == key

    X = X_train[mask].drop(['SPLIT', 'Material', 'SalOrg'], axis='columns').copy()
    y = y_train[mask]['OrderQty'].copy()

    if len(X) == 1:
        X = np.vstack([X, X, X, X, X])
        y = np.vstack([y, y, y, y, y])

    print('{}, {} / {}, X={}, y={}'.format(key, i, len(split_keys), X.shape, y.shape))

    X_fit, X_val, y_fit, y_val = model_selection.train_test_split(X, y, test_size=0.2)

    estimators[key].fit(
        X_fit,
        y_fit,
        gbm__eval_set=[(X_fit, y_fit), (X_val, y_val)],
        gbm__eval_metric=['mae'],
        gbm__early_stopping_rounds=10,
        gbm__verbose=True,
    )

joblib.dump(estimators, 'models/xgboost/estimators.pkl')
