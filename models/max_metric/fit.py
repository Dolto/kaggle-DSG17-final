import pandas as pd
from sklearn import metrics
from sklearn import pipeline
from sklearn.externals import joblib
import xam


# Load data
X_train = pd.read_csv('data/X_train.csv')
y_train = pd.read_csv('data/y_train.csv')['Survived']

# Load the encoder
encoder = joblib.load('models/encoder.pkl')

# Instantiate model
pipe = pipeline.Pipeline([
    ('max_metric', xam.linear_model.ClassificationMetricRegression(metric=metrics.roc_auc_score))
])

# Fit model
pipe.fit(encoder.transform(X_train), y_train)

# Save the fitted model
joblib.dump(pipe, 'models/max_metric/pipeline.pkl')
