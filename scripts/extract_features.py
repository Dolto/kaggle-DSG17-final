import pandas as pd
from sklearn import preprocessing


# Load the training and test set
features = pd.read_csv('data/merged.csv')

# Parse date
features['date'] = pd.to_datetime(features['Month'])
features['month'] = features['date'].dt.month
features['year'] = features['date'].dt.year

# Median quantity ordered per material
features['median_ordered_per_material'] = pd.concat([
    g['OrderQty'].shift().rolling(min_periods=1, window=len(g)).median()
    for _, g in features.groupby('Material')
])

# Mean quantity ordered per material
features['mean_ordered_per_material'] = pd.concat([
    g['OrderQty'].shift().rolling(min_periods=1, window=len(g)).mean()
    for _, g in features.groupby('Material')
])

features.to_csv('data/features.csv', index=False)
