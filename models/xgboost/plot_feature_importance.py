import matplotlib.pyplot as plt
from sklearn.externals import joblib
from xgboost import plot_importance


pipe = joblib.load('models/xgboost/pipeline_best.pkl')
gbm = pipe.steps[-1][1]

plot_importance(gbm)
plt.grid(False)
plt.tight_layout()
plt.savefig('models/xgboost/feature_importance.png', dpi=300)
