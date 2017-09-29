import random

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn import model_selection
from sklearn.externals import joblib


# Load data
X_train = pd.read_csv('data/X_train.csv', sep=';')
y_train = pd.read_csv('data/y_train.csv', sep=';')
X_test = pd.read_csv('data/X_test.csv', sep=';')
y_test = pd.read_csv('data/y_test.csv', sep=';')

# Create a validation set with 20% of the training set
X_fit, X_val, y_fit, y_val = model_selection.train_test_split(X_train, y_train, test_size=0.2)

# Load the tuned pipeline
pipeline = joblib.load('models/xgboost/pipeline.pkl')

predictions = []

n = 10
for i in range(n):
    pipeline.set_params(gbm__seed=random.randint(1, 10e9))
    pipeline.fit(
        X_fit,
        y_fit['OrderQty'],
        gbm__eval_set=[(X_val, y_val['OrderQty'])]
    )
    fit_score = metrics.accuracy_score(y_fit['OrderQty'], pipeline.predict(X_fit))
    val_score = metrics.accuracy_score(y_val['OrderQty'], pipeline.predict(X_val))
    print('Model {} / {}, fit_score: {:.5f}, val_score: {:.5f}'.format(i+1, n, fit_score, val_score))
    predictions.append(pipeline.predict(X_test))

submission = pd.DataFrame(data={
    'id': range(len(y_test)),
    'demand': np.mean(predictions, axis=0)
}).sort_values('id')

submission.to_csv('models/xgboost/submission_xgboost_bagged.csv', index=False)
