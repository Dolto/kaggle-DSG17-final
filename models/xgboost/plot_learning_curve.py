import matplotlib.pyplot as plt
from sklearn.externals import joblib


pipe = joblib.load('models/xgboost/pipeline.pkl')
gbm = pipe.steps[-1][1]

# Retrieve performance metrics
results = gbm.evals_result()
epochs = len(results['validation_0']['auc'])
x_axis = range(0, epochs)

# Plot ROC AUC
fig, ax = plt.subplots()
ax.plot(x_axis, results['validation_0']['auc'], label='Train')
ax.plot(x_axis, results['validation_1']['auc'], label='Test')
ax.legend()
plt.xlabel('Iterations')
plt.ylabel('ROC AUC')
plt.title('XGBoost ROC AUC')
plt.tight_layout()
plt.savefig('models/xgboost/roc_auc_learning_curve.png', dpi=300)
