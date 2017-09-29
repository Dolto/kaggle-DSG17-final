import numpy as np
import pandas as pd


train = pd.read_csv('data/train.csv', sep=';')

train.set_index(pd.to_datetime(train['Month']), inplace=True)

train = train.groupby(['SalOrg', 'Material']).resample('1M').sum()

train['OrderQty'] = train['OrderQty'].fillna(0)

train = train.reset_index()

train.to_csv('data/train_resampled.csv', index=False, sep=';')
