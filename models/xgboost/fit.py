import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn import pipeline
from sklearn import preprocessing
from sklearn.externals import joblib
from splitting import SplittingEstimator
import xgboost as xgb


# Load data
X_train = pd.read_csv('data/X_train.csv', sep=';')
y_train = pd.read_csv('data/y_train.csv', sep=';')['OrderQty']

# Create a validation set with 20% of the training set
X_fit, X_val, y_fit, y_val = model_selection.train_test_split(X_train, y_train, test_size=0.2)

pipe = pipeline.Pipeline([
    ('gbm', xgb.XGBRegressor(
        n_estimators=2000,
        learning_rate=0.05,
        max_depth=13,
        subsample=0.8,
        tree_method='exact'
    ))
])

def split(x):
    return x['']

pipe = SplittingEstimator(pipe, split)

pipe.fit(
    X_fit,
    y_fit,
    gbm__eval_set=[(X_fit, y_fit), (X_val, y_val)],
    gbm__eval_metric=['mae'],
    gbm__early_stopping_rounds=10,
    gbm__verbose=True
)

joblib.dump(pipe, 'models/xgboost/pipeline.pkl')
