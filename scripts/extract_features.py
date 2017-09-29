import pandas as pd
from sklearn import preprocessing


# Load the training and test set
features = pd.read_csv('data/merged.csv', sep=';')

# Fill dates
features['Month'].fillna(features['date'], inplace=True)

# Parse date
features['Month'] = pd.to_datetime(features['Month'])
features['month'] = features['Month'].dt.month
features['year'] = features['Month'].dt.year

# Median quantity ordered per material
features['median_ordered_per_material'] = pd.concat([
    g.sort_values('Month')['OrderQty'].shift().rolling(min_periods=1, window=len(g)).median()
    for _, g in features.groupby('Material')
])
features = features[features['median_ordered_per_material'].notnull()]

# Mean quantity ordered per material
features['mean_ordered_per_material'] = pd.concat([
    g.sort_values('Month')['OrderQty'].shift().rolling(min_periods=1, window=len(g)).mean()
    for _, g in features.groupby('Material')
])
features = features[features['mean_ordered_per_material'].notnull()]

# Add a month splitter for training/testing
features['month_mod'] = features['month'] % 3

# Drop non-features
features.drop(['Month', 'date'], axis='columns', inplace=True)

features.to_csv('data/features.csv', index=False, sep=';')
