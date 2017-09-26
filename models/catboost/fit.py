import json

from catboost import CatBoostClassifier
import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn.externals import joblib


# Load the best params
params = json.load(open('models/catboost/params.json'))

# Instantiate model
model = CatBoostClassifier().set_params(**params)

# Load data
X_train = pd.read_csv('data/X_train.csv')
y_train = pd.read_csv('data/y_train.csv')['Survived']

# Determine categorical features
cat_features = np.where(X_train.dtypes == int)[0]

# Create a validation set with 20% of the training set
X_fit, X_val, y_fit, y_val = model_selection.train_test_split(X_train, y_train, test_size=0.2)

# Fit the model on the training set
model.fit(
    X_fit,
    y_fit,
    eval_set=[X_val, y_val],
    cat_features=cat_features,
    verbose=True
)

# Save the fitted model
joblib.dump(model, 'models/catboost/model.pkl')
