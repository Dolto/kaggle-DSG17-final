import pandas as pd
from sklearn.externals import joblib


X_test = pd.read_csv('data/X_test.csv', sep=';')
y_test = pd.read_csv('data/y_test.csv', sep=';')['OrderQty']

pipe = joblib.load('models/lightgbm/pipeline.pkl')

y_pred = pipe.predict(X_test)

submission = pd.DataFrame(data={
    'id': range(len(y_pred)),
    'demand': y_pred.clip(0)
})

submission.to_csv('models/lightgbm/submission_lightgbm.csv', index=False)
