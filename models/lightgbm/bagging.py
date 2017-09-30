import random

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn import model_selection
from sklearn.externals import joblib


# Load data
X_train = pd.read_csv('data/X_train.csv', sep=';')
y_train = pd.read_csv('data/y_train.csv', sep=';')['OrderQty']
X_test = pd.read_csv('data/X_test.csv', sep=';')
y_test = pd.read_csv('data/y_test.csv', sep=';')['OrderQty']


def custom_train_test_split(X_train, y_train):
    idx_val = (X_train.year == 2017)
    X_val = X_train[idx_val]
    y_val = y_train[idx_val]
    X_fit = X_train[~idx_val]
    y_fit = y_train[~idx_val]
    return X_fit, X_val, y_fit, y_val

X_fit, X_val, y_fit, y_val = custom_train_test_split(X_train, y_train)

# Load the tuned pipeline
pipeline = joblib.load('models/lightgbm/pipeline.pkl')

predictions = []

n = 5
for i in range(n):
    pipeline.set_params(
        gbm__bagging_seed=int(random.randint(1, 10000)),
        gbm__n_estimators=5000,
    )
    pipeline.fit(
        X_fit,
        y_fit,
        gbm__eval_set=[(X_val, y_val)],
        gbm__eval_metric=['l1'],
        gbm__early_stopping_rounds=20,
        gbm__verbose=True
    )
    predictions.append(pipeline.predict(X_test))

submission = pd.DataFrame(data={
    'id': range(len(np.mean(predictions, axis=0))),
    'demand': np.clip(np.mean(predictions, axis=0), 0, None)
})

submission.to_csv('models/lightgbm/submission_lightgbm_bagged.csv', index=False)
