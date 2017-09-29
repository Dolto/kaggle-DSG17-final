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

for _, g in features.groupby(['SalOrg', 'Material']):
    subset = g.sort_values('Month')
    features['last_order_days_ago_per_material_and_org'] = subset['Month'].diff().dt.days
    features['median_ordered_per_material_and_org'] = subset['OrderQty'].shift().rolling(min_periods=1, window=len(g)).median()
    features['mean_ordered_per_material_and_org'] = subset['OrderQty'].shift().rolling(min_periods=1, window=len(g)).mean()
    features['min_ordered_per_material_and_org'] = subset['OrderQty'].shift().rolling(min_periods=1, window=len(g)).min()
    features['max_ordered_per_material_and_org'] = subset['OrderQty'].shift().rolling(min_periods=1, window=len(g)).max()

for _, g in features.groupby(['Material']):
    subset = g.sort_values('Month')
    features['last_order_days_ago_per_material'] = subset['Month'].diff().dt.days
    features['median_ordered_per_material'] = subset['OrderQty'].shift().rolling(min_periods=1, window=len(g)).median()
    features['mean_ordered_per_material'] = subset['OrderQty'].shift().rolling(min_periods=1, window=len(g)).mean()
    features['min_ordered_per_material'] = subset['OrderQty'].shift().rolling(min_periods=1, window=len(g)).min()
    features['max_ordered_per_material'] = subset['OrderQty'].shift().rolling(min_periods=1, window=len(g)).max()


# Median per product
features['material_median'] = pd.concat([
    g['OrderQty'].median()
    for _, g in features.groupby(['Material'])
])

# Mean per material
features['material_mean'] = pd.concat([
    g['OrderQty'].mean()
    for _, g in features.groupby(['Material'])
])


# Remove empty rows
for col in ['last_order_days_ago_per_material',
            'last_order_days_ago_per_material_and_org',
            'median_ordered_per_material',
            'median_ordered_per_material_and_org',
            'mean_ordered_per_material',
            'mean_ordered_per_material_and_org',
            'min_ordered_per_material',
            'min_ordered_per_material_and_org',
            'max_ordered_per_material',
            'max_ordered_per_material_and_org'
            ]:
    features = features[features[col].notnull()]

# Drop non-features
features.drop(['date'], axis='columns', inplace=True)

# Check no test rows have been dropped
assert features['OrderQty'].isnull().sum() == 116028

features.to_csv('data/features.csv', index=False, sep=';')
