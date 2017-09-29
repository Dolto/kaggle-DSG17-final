import pandas as pd
from sklearn.externals import joblib


X_test = pd.read_csv('data/X_test.csv', sep=';')
y_test = pd.read_csv('data/y_test.csv', sep=';')

pipe = joblib.load('models/xgboost/pipeline.pkl')

y_pred = pipe.predict_proba(X_test)

submission = pd.DataFrame(data={
    'id': y_test['ID'].astype(int),
    'demand': y_pred
}).sort_values('id')

submission.to_csv('models/xgboost/submission_xgboost.csv', index=False)
