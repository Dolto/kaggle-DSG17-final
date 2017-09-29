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


# Sort by date
features.sort_values('Month', inplace=True)


def merge_aggregate_features(keys, window, wname):
    gb = features.groupby(keys)

    m = pd.concat([
        g['OrderQty'].shift().rolling(min_periods=1, window=window(g)).mean()
        for _, g in gb
    ])

    features['mean_{}_{}'.format('_'.join(keys), wname)] = m

    print('mean_{}_{}'.format('_'.join(keys), wname))

    m = pd.concat([
        g['OrderQty'].shift().rolling(min_periods=1, window=window(g)).median()
        for _, g in gb
    ])

    features['median_{}_{}'.format('_'.join(keys), wname)] = m

    print('median_{}_{}'.format('_'.join(keys), wname))

    m = pd.concat([
        g['OrderQty'].shift().rolling(min_periods=1, window=window(g)).min()
        for _, g in gb
    ])

    features['min_{}_{}'.format('_'.join(keys), wname)] = m

    print('min_{}_{}'.format('_'.join(keys), wname))

    m = pd.concat([
        g['OrderQty'].shift().rolling(min_periods=1, window=window(g)).max()
        for _, g in gb
    ])

    features['max_{}_{}'.format('_'.join(keys), wname)] = m

    print('max_{}_{}'.format('_'.join(keys), wname))


merge_aggregate_features(['Material'], lambda x: len(x), 't')
merge_aggregate_features(['Material', 'SalOrg'], lambda x: len(x), 't')
merge_aggregate_features(['Material'], lambda x: 3, 3)
merge_aggregate_features(['Material', 'SalOrg'], lambda x: 3, 3)
merge_aggregate_features(['Material'], lambda x: 5, 5)
merge_aggregate_features(['Material', 'SalOrg'], lambda x: 5, 5)
merge_aggregate_features(['Material'], lambda x: 7, 7)
merge_aggregate_features(['Material', 'SalOrg'], lambda x: 7, 7)

# Median per product
#features['material_median'] = features.groupby('Material').median()

# Mean per material
#features['material_mean'] = features.groupby('Material').mean()

# Remove empty rows
for col in features.columns:
    if col not in ['ID', 'OrderQty', 'date']:
        features = features[features[col].notnull()]

# Drop non-features
features.drop(['date'], axis='columns', inplace=True)

# Check no test rows have been dropped
assert features['OrderQty'].isnull().sum() == 116028

features.to_csv('data/features.csv', index=False, sep=';')
