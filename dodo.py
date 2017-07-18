def task_merge_train_test():
    return {
        'file_dep': [
            'data/provided/train.csv',
            'data/provided/test.csv',
        ],
        'actions': ['python scripts/merge_train_test.py'],
        'targets': ['merged.csv']
    }


def task_extract_features():
    return {
        'file_dep': ['data/merged.csv'],
        'actions': ['python scripts/extract_features.py'],
        'targets': ['data/train_features.csv']
    }


TRAINING_SETS = [
    'data/X_train.csv',
    'data/y_train.csv'
]

TEST_SETS = [
    'data/X_test.csv'
    'data/y_test.csv'
]


def task_split_train_test():
    return {
        'file_dep': ['data/features.csv'],
        'actions': ['python scripts/split_train_test.py'],
        'targets': TRAINING_SETS + TEST_SETS
    }


def task_make_cv():
    return {
        'file_dep': TRAINING_SETS + TEST_SETS,
        'actions': ['python scripts/make_cv.py'],
        'targets': 'data/cv.pkl'
    }


def task_fit_max():
    return {
        'file_dep': TRAINING_SETS + ['data/cv.pkl'],
        'actions': ['python models/max/fit.py'],
        'targets': ['models/max/pipeline.pkl'],
        'verbosity': 2 # To display training progress
    }


def task_predict_max():
    return {
        'file_dep': TEST_SETS + ['models/max/pipeline.pkl'],
        'actions': ['python models/xgboost/predict.py'],
        'targets': ['models/xgboost/submission_max.csv']
    }
