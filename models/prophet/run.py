import pandas as pd


# Load data
y_train = pd.read_csv('data/y_train.csv', sep=';')['OrderQty']
