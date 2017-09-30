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

n = 5
for i in range(n):
    pipeline.set_params(
        gbm__seed=int(random.randint(1, 10000)),
        gbm__n_estimators=2000,
        gbm__learning_rate=0.01,
        gbm__max_depth=8
    )
    pipeline.fit(
        X_fit,
        y_fit['OrderQty'],
        gbm__eval_set=[(X_val, y_val['OrderQty'])],
        gbm__eval_metric=['mae'],
        gbm__early_stopping_rounds=10,
        gbm__verbose=True
    )
    predictions.append(pipeline.predict(X_test))

submission = pd.DataFrame(data={
    'id': range(len(y_test)),
    'demand': np.mean(predictions, axis=0)
}).sort_values('id')

submission.to_csv('models/xgboost/submission_xgboost_bagged.csv', index=False)
