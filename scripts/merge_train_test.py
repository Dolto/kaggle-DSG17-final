import os
import numpy as np
import pandas as pd

SAMPLE = False

train_filepath = os.path.join('data', 'train.csv') if not SAMPLE else os.path.join('data', 'train_sample.csv')
test_filepath = os.path.join('data', 'test.csv') if not SAMPLE else os.path.join('data', 'test_sample.csv')

train = pd.read_csv(train_filepath, sep=';')
test = pd.read_csv(test_filepath, sep=';')
test['is_test'] = True

train = train.groupby(['SalOrg', 'Material', 'Month'])[['OrderQty']].sum().reset_index()

merged = pd.concat((train, test), axis='rows')
merged['is_test'].fillna(False, inplace=True)
merged['Month'].fillna(merged['date'], inplace=True)
merged.drop('date', axis='columns', inplace=True)
merged.set_index(pd.to_datetime(merged['Month']), inplace=True)


merged = merged.groupby(['SalOrg', 'Material']).resample('1M').sum()
merged['OrderQty'][merged['is_test'].isnull()] = 0
merged.reset_index(inplace=True)
merged.drop('is_test', axis='columns', inplace=True)

merged.to_csv('data/merged.csv', index=False, sep=';')
