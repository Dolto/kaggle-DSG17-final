import numpy as np
import pandas as pd

NB_OBS = 200
np.random.seed(42)

train_df = pd.read_csv('data/train.csv', sep=';')

material_ids = train_df['Material'].unique()
mids_sample = np.random.choice(material_ids, NB_OBS)

train_df = train_df[train_df['Material'].isin(mids_sample)]
train_columns = [
    'ID', 'First_MAD', 'SalOrg', 'DC', 'Ship_To', 'ordre', 'Plant',
    'Material', 'ItemCat', 'OrderQty', 'LT', 'LogABC', 'MOQ', 'ROP',
    'SafetyStk', 'PL', 'MktABC', 'SubFct', 'Gross_Weight', 'Length',
    'Width', 'Height', 'Volume', 'Gamma', 'Manufacturer', 'Business',
    'Month', 'CBO_CBO_Qty_Shortage', 'Age_ZN_ZI_years', 'DP_FAMILY_CODE',
    'PRODUCT_STATUS', 'ORIGINAL_SUPPLIER', 'SUBRANGE',
    'Comp_reference_number', 'Name_Of_Competitor', 'COMP_PRICE_MIN',
    'COMP_PRICE_AVG', 'COMP_PRICE_MAX', 'PRICE', 'NEAREST_COMP_PRICE_MIN',
    'NEAREST_COMP_PRICE_MAX'
]
assert np.all(train_df.columns == train_columns)
train_df.to_csv('data/train_sample.csv', sep=';', index=False)

test_df = pd.read_csv('data/test.csv', sep=',')
test_df = test_df[test_df['Material'].isin(mids_sample)]
test_columns = ['ID', 'SalOrg', 'Material', 'date']
assert np.all(test_df.columns == test_columns)
test_df.to_csv('data/test_sample.csv', sep=';', index=False)

