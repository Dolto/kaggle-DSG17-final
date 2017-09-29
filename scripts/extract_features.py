import pandas as pd
from sklearn import preprocessing


# Load the training and test set
features = pd.read_csv('data/merged.csv')

# Parse date
features['date'] = pd.to_datetime(features['Month'])
features['month'] = features['date'].dt.month
features['year'] = features['date'].dt.year

# Median quantity ordered per material
median_qts = sample.groupby('Material')['OrderQty'].median().to_frame('median_OrderQty_material').reset_index()
features = pd.merge(left=features, right=meds, on='Material')

features.to_csv('data/features.csv', index=False)
