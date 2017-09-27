import numpy as np
import pandas as pd
from sklearn.externals import joblib


# Load the fitted pipeline
pipeline = joblib.load('models/auc_regressor/pipeline.pkl')

# Load the encoder
encoder = joblib.load('models/encoder.pkl')

# Load data
X_test = pd.read_csv('data/X_test.csv')
y_test = pd.read_csv('data/y_test.csv')

# Fit the pipeline on the training set
y_pred = np.rint(pipeline.predict(encoder.transform(X_test))).astype(int) + 1

# Create submission
submission = pd.DataFrame({'PassengerId': y_test['PassengerId'].astype(int), 'Survived': y_pred})
submission.to_csv('models/auc_regressor/submission_auc_regressor.csv', index=False)
