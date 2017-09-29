import os
import numpy as np
import pandas as pd

SAMPLE = False

train_filepath = os.path.join('data', 'train.csv') if not SAMPLE else os.path.join('data', 'train_sample.csv')
test_filepath = os.path.join('data', 'test.csv') if not SAMPLE else os.path.join('data', 'test_sample.csv')

train = pd.read_csv(train_filepath, sep=';')
test = pd.read_csv(test_filepath, sep=';')

train = train.groupby(['SalOrg', 'Material', 'Month'])['OrderQty', 'ID'].sum().reset_index()

merged = pd.concat((train, test), axis='rows')

assert np.all(merged.columns == ['ID', 'Material', 'Month', 'OrderQty', 'SalOrg', 'date'])
merged.to_csv('data/merged.csv', index=False, sep=';')
