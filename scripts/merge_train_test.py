import pandas as pd


train = pd.read_csv('data/train.csv', sep=';')
test = pd.read_csv('data/test.csv')

train = train[['ID', 'SalOrg', 'Material', 'Month']]

merged = pd.concat((train, test), axis='rows')
merged.to_csv('data/merged.csv', index=False)
