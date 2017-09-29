import pandas as pd
from sklearn.externals import joblib


X_test = pd.read_csv('data/X_test.csv', sep=';')
y_test = pd.read_csv('data/y_test.csv', sep=';')

X_test['Material'] = y_test['Material'].copy()
X_test['SalOrg'] = y_test['SalOrg'].copy()
X_test['SPLIT'] = X_test['Material']

estimators = joblib.load('models/xgboost/estimators.pkl')

for split, estimator in estimators.items():
    mask = X_test['SPLIT'] == split
    y_pred = estimator.predict(X_test[mask].drop(['SPLIT', 'Material', 'SalOrg'], axis='columns'))
    y_test['OrderQty'][mask] = y_pred

submission = pd.DataFrame(data={
    'id': range(len(y_test)),
    'demand': y_test['OrderQty'].clip(0)
})

submission.to_csv('models/xgboost/submission_xgboost_split.csv', index=False)
