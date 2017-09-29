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

# Last order days ago per material
features['last_order_days_ago_per_material'] = pd.concat([
    g.sort_values('Month')['Month'].diff().dt.days
    for _, g in features.groupby(['SalOrg', 'Material'])
])

# Last order days ago per material and per org
features['last_order_days_ago_per_material_and_org'] = pd.concat([
    g.sort_values('Month')['Month'].diff().dt.days
    for _, g in features.groupby(['SalOrg', 'Material'])
])

# Median quantity ordered per material and per org
features['median_ordered_per_material_and_org'] = pd.concat([
    g.sort_values('Month')['OrderQty'].shift().rolling(min_periods=1, window=len(g)).median()
    for _, g in features.groupby(['Material', 'SalOrg'])
])

# Median quantity ordered per material
features['median_ordered_per_material'] = pd.concat([
    g.sort_values('Month')['OrderQty'].shift().rolling(min_periods=1, window=len(g)).median()
    for _, g in features.groupby('Material')
])

# Add a month splitter for training/testing
features['month_mod'] = features['month'] % 3

# Average per mod
# for i in range(3):
#     subset = features[features['month_mod'] == i]
#     features['previous_ordered_mod'] = pd.concat([
#         g.sort_values('Month')['OrderQty'].shift(1)
#         for _, g in subset.groupby('Material')
#     ])


# Remove empty rows
for feature in ['median_ordered_per_material', 'last_order_days_ago_per_material',
                'last_order_days_ago_per_material_and_org']:
    features = features[features[feature].notnull()]

# Drop non-features
features.drop(['Month', 'date'], axis='columns', inplace=True)

features.to_csv('data/features.csv', index=False, sep=';')
