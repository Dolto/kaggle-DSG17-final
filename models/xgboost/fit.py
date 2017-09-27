import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn import pipeline
from sklearn import preprocessing
from sklearn.externals import joblib
import xgboost as xgb


# Load the encoder
encoder = joblib.load('models/encoder.pkl')

# Load data
X_train = encoder.transform(pd.read_csv('data/X_train.csv'))
y_train = pd.read_csv('data/y_train.csv')['Survived']

# Create a validation set with 20% of the training set
X_train, X_val, y_train, y_val = model_selection.train_test_split(X_train, y_train, test_size=0.2)

pipe = pipeline.Pipeline([
    ('gbm', xgb.XGBClassifier(
        n_estimators=10000,
        learning_rate=0.007,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=sum(y_train == 0) / sum(y_train == 1)
    ))
])

pipe.fit(
    X_train,
    y_train,
    gbm__eval_set=[(X_train, y_train), (X_val, y_val)],
    gbm__eval_metric=['auc'],
    gbm__early_stopping_rounds=10,
    gbm__verbose=True
)

joblib.dump(pipe, 'models/xgboost/pipeline.pkl')
