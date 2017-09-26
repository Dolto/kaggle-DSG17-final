import matplotlib.pyplot as plt
import pandas as pd
from sklearn.externals import joblib
import seaborn as sns


# Load model
model = joblib.load('models/catboost/model.pkl')

# Get the list of feature names
X_test = pd.read_csv('data/X_test.csv')

features = pd.DataFrame({'name': X_test.columns, 'importance': model.feature_importances_})


ax = sns.barplot(
    x='importance',
    y='name',
    data=features.sort_values('importance', ascending=False)
)

ax.set_xlabel('Feature importance')
ax.set_ylabel('Feature name')

plt.tight_layout()
plt.show()
