import numpy as np
import pandas as pd


submission_files = [
    'models/lightgbm/submission_xgboost.csv',
    'models/lightgbm/submission_xgboost 2.csv'
]

submissions = [pd.read_csv(sub_file) for sub_file in submission_files]

average_pred = np.mean([sub['demand'] for sub in submissions], axis=0)

blended_submission = submissions[0]
blended_submission['demand'] = average_pred
blended_submission.to_csv('models/blended_submission.csv', index=False)
