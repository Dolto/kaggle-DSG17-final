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



true_test = pd.read_csv('data/test.csv', sep=';')
test = features[~train_idxs].copy()

true_test['sort_col'] = true_test['date'] + true_test['SalOrg'].str.upper() + true_test['Material'].str.upper()
test['Month'] = pd.to_datetime(test['Month'])
test['Month'] = test['Month'].apply(lambda x: x.strftime('%Y-%m'))
test['sort_col'] = test['Month'] + test['SalOrg'].str.upper() + test['Material'].str.upper()

test.set_index('sort_col', inplace=True)
true_test.set_index('sort_col', inplace=True)
test = test.align(true_test, axis=0)[0]

X_test = test.drop(non_features, axis='columns')
X_test.to_csv('data/X_test.csv', index=False, sep=';')
print('X_test saved')

y_test = test[non_features]
y_test.to_csv('data/y_test.csv', index=False, sep=';')
print('y_test saved')
