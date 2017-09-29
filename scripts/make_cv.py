import datetime as dt

from sklearn import model_selection
from sklearn.externals import joblib
import xam


cv = xam.model_selection.DatetimeCV(timedelta=dt.timedelta(days=30.5))
joblib.dump(cv, 'models/cv.pkl')
