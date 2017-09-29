import pandas as pd

data = pd.read_csv('data/train.csv', sep=';')

features_names = [
    'ItemCat',
    'LogABC',
    'PL',
    'MktABC',
    'SubFct',
    'Gamma',
    'DP_FAMILY_CODE',
    'PRODUCT_STATUS'
]

mask = ['Material'] + features_names 
mapping = data[mask].groupby('Material').first()
mapping.reset_index(inplace=True)

features = pd.read_csv('data/features.csv', sep=';')

features_cat = features.merge(mapping, on='Material', how='inner')
features_encoded = pd.get_dummies(features_cat, columns=features_names)
features_encoded.to_csv('data/features_cat.csv', sep=';', index=False)
