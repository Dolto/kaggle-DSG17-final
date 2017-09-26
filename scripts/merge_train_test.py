import pandas as pd


train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

data = pd.concat((train, test), axis='rows')
data.to_csv('data/merged.csv', index=False)
