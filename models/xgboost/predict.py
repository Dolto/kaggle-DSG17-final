import pandas as pd
from sklearn.externals import joblib


X_test = pd.read_csv('data/X_test.csv')
y_test = pd.read_csv('data/y_test.csv')

pipe = joblib.load('models/xgboost/pipeline.pkl')

# Load the encoder
encoder = joblib.load('models/encoder.pkl')

y_pred = pipe.predict_proba(encoder.transform(X_test))[:, 1]

submission = pd.DataFrame(data={
    'PassengerId': y_test['PassengerId'].astype(int),
    'Survived': y_pred
}).sort_values('PassengerId')

submission.to_csv('models/xgboost/submission_xgboost.csv', index=False)
