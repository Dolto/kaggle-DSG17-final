import json

from catboost import CatBoostRegressor
import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn import pipeline
from sklearn.externals import joblib


X_train = pd.read_csv('data/X_train.csv', sep=';')
y_train = pd.read_csv('data/y_train.csv', sep=';')['OrderQty']

# Create a validation set with 20% of the training set
X_fit, X_val, y_fit, y_val = model_selection.train_test_split(X_train, y_train, test_size=0.2)


# Fit the model on the training set
pipe = CatBoostRegressor(
    iterations=2000,
    loss_function='MAE'
)

pipe.fit(
    X_fit,
    y_fit,
    eval_set=[X_val, y_val],
    verbose=True
)

# Save the fitted model
joblib.dump(pipe, 'models/catboost/pipeline.pkl')
