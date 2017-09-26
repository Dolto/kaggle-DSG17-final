from sklearn import model_selection
from sklearn.externals import joblib


cv = model_selection.StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
joblib.dump(cv, 'models/cv.pkl')
