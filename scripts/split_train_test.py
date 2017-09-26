import pandas as pd


features = pd.read_csv('data/features.csv')

train_idxs = features['Survived'].notnull()

train = features[train_idxs]
test = features[~train_idxs]

train.drop(['Survived', 'PassengerId'], axis='columns').to_csv('data/X_train.csv', index=False)
train[['Survived', 'PassengerId']].to_csv('data/y_train.csv', index=False)

test.drop(['Survived', 'PassengerId'], axis='columns').to_csv('data/X_test.csv', index=False)
test[['Survived', 'PassengerId']].to_csv('data/y_test.csv', index=False)
