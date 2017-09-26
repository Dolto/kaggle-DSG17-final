import json

from catboost import CatBoostClassifier
import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn import pipeline
from sklearn.externals import joblib


# Load data
X_train = pd.read_csv('data/X_train.csv')
y_train = pd.read_csv('data/y_train.csv')['Survived']

# Determine categorical features
cat_features = np.where(X_train.dtypes == int)[0]

print(cat_features)

# Load CV
cv = joblib.load('models/cv.pkl')

# Create a validation set with 20% of the training set
X_fit, X_val, y_fit, y_val = model_selection.train_test_split(X_train, y_train, test_size=0.2)

# Instantiate grid search
grid = model_selection.GridSearchCV(
    estimator=CatBoostClassifier(
        iterations=2000,
        od_type='Iter',
        od_wait=20,
        use_best_model=True,
        eval_metric='Logloss'
    ),
    param_grid=[
        {
            'depth': [6],
            'learning_rate': [0.03],
            'l2_leaf_reg': [3],
        },
    ],
    cv=cv,
    scoring='accuracy',
    fit_params={
        'eval_set': [X_val, y_val],
        'cat_features': cat_features
    },
    verbose=2
)

# Perform the grid search
grid.fit(X_fit, y_fit)

# Save the best params
with open('models/catboost/params.json', 'w') as outfile:
    json.dump(grid.best_estimator_.get_params(), outfile)
