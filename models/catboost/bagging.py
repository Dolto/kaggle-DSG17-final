import random

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn import model_selection
from sklearn.externals import joblib


# Load data
X_train = pd.read_csv('data/X_train.csv')
y_train = pd.read_csv('data/y_train.csv')
X_test = pd.read_csv('data/X_test.csv')
y_test = pd.read_csv('data/y_test.csv')

# Determine categorical features
cat_features = np.where(X_train.dtypes == int)[0]

# Create a validation set with 20% of the training set
X_fit, X_val, y_fit, y_val = model_selection.train_test_split(X_train, y_train, test_size=0.2)

# Load the tuned model
model = joblib.load('models/catboost/model.pkl')

predictions = []

n = 10
for i in range(n):
    model.set_params(random_seed=random.randint(1, 10e9))
    model.fit(
        X_fit,
        y_fit['Survived'],
        eval_set=[X_val, y_val['Survived']],
        cat_features=cat_features
    )
    fit_score = metrics.accuracy_score(y_fit['Survived'], model.predict(X_fit))
    val_score = metrics.accuracy_score(y_val['Survived'], model.predict(X_val))
    print('Model {} / {}, fit_score: {:.5f}, val_score: {:.5f}'.format(i+1, n, fit_score, val_score))
    predictions.append(model.predict_proba(X_test)[:, 1])


y_pred = np.rint(np.mean(predictions, axis=0)).astype(int)

submission = pd.DataFrame({'PassengerId': y_test['PassengerId'], 'Survived': y_pred})
submission.to_csv('models/catboost/submission_catboost_bagged.csv', index=False)
