import numpy as np
import pandas as pd


submission_files = [
    'models/xgboost/submission_xgboost.csv',
    'models/catboost/submission_catboost.csv'
    'models/catboost/submission_catboost_bagged.csv'
]

submissions = [pd.read_csv(sub_file) for sub_file in submission_files]

average_pred = np.mean([sub['Survived'] for sub in submissions])

blended_submission = submissions[0]
blended_submission['Survived'] = average_pred
blended_submission.to_csv('models/blended_submission.csv', index=False)
