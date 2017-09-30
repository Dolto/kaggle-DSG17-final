import pandas as pd


# Load the training and test set
features = pd.read_csv('data/merged.csv', sep=';')

# Parse date
features['Month'] = pd.to_datetime(features['Month'])
features['month'] = features['Month'].dt.month
features['year'] = features['Month'].dt.year


# Sort by date
features.sort_values('Month', inplace=True)


def merge_aggregate_features(keys, window, wname):
    gb = features.groupby(keys)

    m = pd.concat([
        g['OrderQty'].shift().rolling(min_periods=0, window=window(g)).mean()
        for _, g in gb
    ])
    features['mean_{}_{}'.format('_'.join(keys), wname)] = m
    print('mean_{}_{}'.format('_'.join(keys), wname))

    m = pd.concat([
        g['OrderQty'].shift().rolling(min_periods=0, window=window(g)).median()
        for _, g in gb
    ])
    features['median_{}_{}'.format('_'.join(keys), wname)] = m
    print('median_{}_{}'.format('_'.join(keys), wname))

    m = pd.concat([
        g['OrderQty'].shift().rolling(min_periods=0, window=window(g)).min()
        for _, g in gb
    ])
    features['min_{}_{}'.format('_'.join(keys), wname)] = m
    print('min_{}_{}'.format('_'.join(keys), wname))

    m = pd.concat([
        g['OrderQty'].shift().rolling(min_periods=0, window=window(g)).max()
        for _, g in gb
    ])
    features['max_{}_{}'.format('_'.join(keys), wname)] = m
    print('max_{}_{}'.format('_'.join(keys), wname))


merge_aggregate_features(['Material'], lambda x: len(x), 't')
merge_aggregate_features(['Material', 'SalOrg'], lambda x: len(x), 't')
#merge_aggregate_features(['Material'], lambda x: 3, 3)
merge_aggregate_features(['Material', 'SalOrg'], lambda x: 3, 3)
#merge_aggregate_features(['Material'], lambda x: 5, 5)
merge_aggregate_features(['Material', 'SalOrg'], lambda x: 5, 5)
#merge_aggregate_features(['Material'], lambda x: 7, 7)
merge_aggregate_features(['Material', 'SalOrg'], lambda x: 7, 7)



medians = features.groupby('Material')['OrderQty'].median().to_frame('material_median').reset_index()
features = pd.merge(features, medians, on='Material')

means = features.groupby('Material')['OrderQty'].mean().to_frame('material_mean').reset_index()
features = pd.merge(features, means, on='Material')


for col in features.columns:
    if col.endswith('Material_3') or col.endswith('Material_5') or col.endswith('Material_7'):
        features.drop(col, axis='columns', inplace=True)



features.isnull().sum()
features[features['OrderQty'].isnull()].isnull().sum()


# Remove empty rows
features_2 = features.copy()
for col in features_2.columns:
    if col not in ['ID', 'OrderQty', 'date', 'n_row']:
        features_2 = features_2[features_2[col].notnull()]

# Check no test rows have been dropped
assert features_2['OrderQty'].isnull().sum() == 116028

features_2.to_csv('data/features.csv', index=False, sep=';')
