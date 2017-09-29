import pandas as pd


features = pd.read_csv('data/features.csv')

train_idxs = features['OrderQty'].notnull()

train = features[train_idxs]
test = features[~train_idxs]

train.drop(['OrderQty', 'ID'], axis='columns').to_csv('data/X_train.csv', index=False)
train[['OrderQty', 'ID']].to_csv('data/y_train.csv', index=False)

test.drop(['OrderQty', 'ID'], axis='columns').to_csv('data/X_test.csv', index=False)
test[['OrderQty', 'ID']].to_csv('data/y_test.csv', index=False)
