import pandas as pd


features = pd.read_csv('data/features.csv', sep=';')

train_idxs = features['OrderQty'].notnull()

train = features[train_idxs]

non_features = ['OrderQty', 'Material', 'SalOrg', 'Month', 'ID']

X_train = train.drop(non_features, axis='columns')
X_train.to_csv('data/X_train.csv', index=False, sep=';')
print('X_train saved')

y_train = train[non_features]
y_train.to_csv('data/y_train.csv', index=False, sep=';')
print('y_train saved')

test = features[~train_idxs]

X_test = test.drop(non_features, axis='columns')
X_test.to_csv('data/X_test.csv', index=False, sep=';')
print('X_test saved')

y_test = test[non_features]
y_test.to_csv('data/y_test.csv', index=False, sep=';')
print('y_test saved')
