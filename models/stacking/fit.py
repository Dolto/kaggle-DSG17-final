import catboost
import numpy as np
import pandas as pd
from sklearn import ensemble
from sklearn import linear_model
from sklearn import model_selection
from sklearn.externals import joblib
import xam


# Load the encoder
encoder = joblib.load('models/encoder.pkl')

# Load data
X_train = encoder.transform(pd.read_csv('data/X_train.csv'))
y_train = pd.read_csv('data/y_train.csv')['Survived']

# Determine categorical features
cat_features = np.where(X_train.dtypes == int)[0]

# Create a validation set with 20% of the training set
X_fit, X_val, y_fit, y_val = model_selection.train_test_split(X_train, y_train, test_size=0.2)

# Initialize stage 0 models
models = {
    'rf': ensemble.RandomForestClassifier(random_state=1),
    'catboost': catboost.CatBoostClassifier()
}

# Initialize stack
stack = xam.stacking.StackingClassifier(
    models=models,
    meta_model=linear_model.LogisticRegression(),
    cv=model_selection.StratifiedKFold(n_splits=10),
    use_base_features=True,
    use_proba=True
)

stack.fit(
    X=X_train,
    y=y_train,
    fit_params={
        'catboost': {
            'eval_set': [X_val, y_val],
            'verbose': True
        }
    }
)

# Save the fitted model
joblib.dump(stack, 'models/stacking/stack.pkl')
