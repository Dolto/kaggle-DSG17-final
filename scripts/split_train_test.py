import pandas as pd

features = pd.read_csv('data/features_cat.csv', sep=';')

train_idxs = features['OrderQty'].notnull()

train = features[train_idxs]

X_train = train.drop(['OrderQty', 'ID', 'Material', 'SalOrg'], axis='columns')
X_train.to_csv('data/X_train.csv', index=False, sep=';')
y_train = train[['OrderQty', 'Material', 'SalOrg', 'ID']]
y_train.to_csv('data/y_train.csv', index=False, sep=';')

test = features[~train_idxs]

X_test = test.drop(['OrderQty', 'ID', 'Material', 'SalOrg'], axis='columns')
X_test.to_csv('data/X_test.csv', index=False, sep=';')
y_test = test[['OrderQty', 'Material', 'SalOrg', 'ID']]
y_test.to_csv('data/y_test.csv', index=False, sep=';')
