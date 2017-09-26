import category_encoders as ce
import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn.externals import joblib

# Load data
X_train = pd.read_csv('data/X_train.csv')
X_test = pd.read_csv('data/X_test.csv')

# Determine categorical features
cat_features = list(X_train.columns[np.where(X_train.dtypes == int)[0]])

# Fit and save the encoder
encoder = ce.OneHotEncoder(cols=cat_features).fit(pd.concat((X_train, X_test), axis='rows'))
joblib.dump(encoder, 'models/encoder.pkl')
