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

# Median quantity ordered per material
features['median_ordered_per_material'] = pd.concat([
    g.sort_values('Month')['OrderQty'].shift().rolling(min_periods=1, window=len(g)).median()
    for _, g in features.groupby('Material')
])

# Median quantity ordered per material and per org
features['median_ordered_per_material_and_org'] = pd.concat([
    g.sort_values('Month')['OrderQty'].shift().rolling(min_periods=1, window=len(g)).median()
    for _, g in features.groupby(['Material', 'SalOrg'])
])

# Mean quantity ordered per material
features['mean_ordered_per_material'] = pd.concat([
    g.sort_values('Month')['OrderQty'].shift().rolling(min_periods=0, window=len(g)).mean()
    for _, g in features.groupby('Material')
])

# Mean quantity ordered per material and per org
features['mean_ordered_per_material_and_org'] = pd.concat([
    g.sort_values('Month')['OrderQty'].shift().rolling(min_periods=0, window=len(g)).mean()
    for _, g in features.groupby(['Material', 'SalOrg'])
])

# # Standard dev quantity ordered per material
# features['std_ordered_per_material'] = pd.concat([
#     g.sort_values('Month')['OrderQty'].shift().rolling(min_periods=1, window=len(g)).std()
#     for _, g in features.groupby('Material')
# ])

# # Standard dev quantity ordered per material and per org
# features['std_ordered_per_material_and_org'] = pd.concat([
#     g.sort_values('Month')['OrderQty'].shift().rolling(min_periods=1, window=len(g)).std()
#     for _, g in features.groupby(['Material', 'SalOrg'])
# ])

# # Min quantity ordered per material
# features['min_ordered_per_material'] = pd.concat([
#     g.sort_values('Month')['OrderQty'].shift().rolling(min_periods=0, window=len(g)).min()
#     for _, g in features.groupby('Material')
# ])

# # Min quantity ordered per material and per org
# features['min_ordered_per_material_and_org'] = pd.concat([
#     g.sort_values('Month')['OrderQty'].shift().rolling(min_periods=0, window=len(g)).min()
#     for _, g in features.groupby(['Material', 'SalOrg'])
# ])

# # Max quantity ordered per material
# features['max_ordered_per_material'] = pd.concat([
#     g.sort_values('Month')['OrderQty'].shift().rolling(min_periods=0, window=len(g)).max()
#     for _, g in features.groupby('Material')
# ])

# # Max quantity ordered per material and per org
# features['max_ordered_per_material_and_org'] = pd.concat([
#     g.sort_values('Month')['OrderQty'].shift().rolling(min_periods=0, window=len(g)).max()
#     for _, g in features.groupby(['Material', 'SalOrg'])
# ])


# Add a month splitter for training/testing
features['month_mod'] = features['month'] % 3

# Per mod
# for i in range(3):
#     subset = features[features['month_mod'] == i]
    # features['previous_ordered_mod'] = pd.concat([
    #     g.sort_values('Month')['OrderQty'].shift(1)
    #     for _, g in subset.groupby('Material')
    # ])

# Remove empty rows
for col in ['last_order_days_ago_per_material',
            'last_order_days_ago_per_material_and_org',
            'median_ordered_per_material',
            'median_ordered_per_material_and_org',
            'mean_ordered_per_material',
            'mean_ordered_per_material_and_org'#,
            # 'std_ordered_per_material',
            # 'std_ordered_per_material_and_org',
            # 'min_ordered_per_material',
            # 'min_ordered_per_material_and_org',
            # 'max_ordered_per_material',
            # 'max_ordered_per_material_and_org'
            ]:
    features = features[features[col].notnull()]

# Drop non-features
features.drop(['date'], axis='columns', inplace=True)

# Check no test rows have been dropped
assert features['OrderQty'].isnull().sum() == 116028

features.to_csv('data/features.csv', index=False, sep=';')
