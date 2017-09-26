import numpy as np
import pandas as pd
from sklearn.externals import joblib


# Load the fitted pipeline
pipeline = joblib.load('models/catboost/model.pkl')

# Load data
X_test = pd.read_csv('data/X_test.csv')
y_test = pd.read_csv('data/y_test.csv')

# Fit the pipeline on the training set
y_pred = np.rint(pipeline.predict(X_test)).astype(int)

# Create submission
submission = pd.DataFrame({'PassengerId': y_test['PassengerId'], 'Survived': y_pred})
submission.to_csv('models/catboost/submission_catboost.csv', index=False)
