def task_sample():
    return {
        'actions': ['python scripts/sample.py'],
        'file_dep': [
            'data/train.csv',
            'data/test.csv'
        ],
        'targets': [
            'data/train_sample.csv',
            'data/test_sample.csv',
        ]
    }

def task_merge_train_test():
    return {
        'actions': ['python scripts/merge_train_test.py'],
        'file_dep': [
            'data/train.csv',
            'data/test.csv',
        ],
        'targets': ['data/merged.csv']
    }

def task_extract_features():
    return {
        'actions': ['python scripts/extract_features.py'],
        'file_dep': ['data/merged.csv'],
        'targets': ['data/features.csv']
    }


TRAINING_SETS = [
    'data/X_train.csv',
    'data/y_train.csv'
]

TEST_SETS = [
    'data/X_test.csv',
    'data/y_test.csv'
]


def task_split_train_test():
    return {
        'actions': ['python scripts/split_train_test.py'],
        'file_dep': ['data/features.csv'],
        'targets': TRAINING_SETS + TEST_SETS
    }


def task_make_cv():
    return {
        'actions': ['python scripts/make_cv.py'],
        'file_dep': TRAINING_SETS + TEST_SETS,
        'targets': ['models/cv.pkl']
    }


def task_grid_search_catboost():
    return {
        'actions': ['python models/catboost/grid_search.py'],
        'file_dep': TRAINING_SETS + ['models/cv.pkl'],
        'targets': ['models/catboost/params.json'],
        'verbosity': 2
    }


def task_fit_catboost():
    return {
        'actions': ['python models/catboost/fit.py'],
        'file_dep': TRAINING_SETS + ['models/catboost/params.json'],
        'targets': ['models/catboost/model.pkl'],
        'verbosity': 2
    }


def task_predict_catboost():
    return {
        'actions': ['python models/catboost/predict.py'],
        'file_dep': TEST_SETS + ['models/catboost/model.pkl'],
        'targets': ['models/catboost/submission_catboost.csv']
    }


def task_bagging_catboost():
    return {
        'actions': ['python models/catboost/bagging.py'],
        'file_dep': TEST_SETS + ['models/catboost/model.pkl'],
        'targets': ['models/catboost/submission_catboost_bagged.csv'],
        'verbosity': 2
    }


def task_fit_xgboost():
    return {
        'actions': ['python models/xgboost/fit.py'],
        'file_dep': TRAINING_SETS + ['models/encoder.pkl'],
        'targets': ['models/xgboost/pipeline.pkl'],
        'verbosity': 2
    }


def task_predict_xgboost():
    return {
        'actions': ['python models/xgboost/predict.py'],
        'file_dep': TEST_SETS + ['models/encoder.pkl', 'models/xgboost/pipeline.pkl'],
        'targets': ['models/xgboost/submission_xgboost.csv']
    }


def task_plot_xgboost():
    return {
        'actions': [
            'python models/xgboost/plot_learning_curve.py',
            'python models/xgboost/plot_feature_importance.py',
        ],
        'file_dep': ['models/xgboost/pipeline.pkl'],
        'targets': [
            'models/xgboost/roc_auc_learning_curve.png',
            'models/xgboost/feature_importance.png'
        ]
    }


def task_fit_auc_regressor():
    return {
        'actions': ['python models/auc_regressor/fit.py'],
        'file_dep': TRAINING_SETS + ['models/encoder.pkl'],
        'targets': ['models/auc_regressor/pipeline.pkl'],
        'verbosity': 2
    }


def task_predict_auc_regressor():
    return {
        'actions': ['python models/auc_regressor/predict.py'],
        'file_dep': TEST_SETS + ['models/encoder.pkl', 'models/auc_regressor/pipeline.pkl'],
        'targets': ['models/auc_regressor/submission_auc_regressor.csv']
    }


def task_fit_stacking():
    return {
        'actions': ['python models/stacking/fit.py'],
        'file_dep': TRAINING_SETS + ['models/encoder.pkl'],
        'targets': ['models/stacking/stack.pkl'],
        'verbosity': 2
    }


def task_predict_stacking():
    return {
        'actions': ['python models/stacking/predict.py'],
        'file_dep': TEST_SETS + ['models/encoder.pkl', 'models/stacking/stack.pkl'],
        'targets': ['models/stacking/submission_stacking.csv']
    }


def task_blend_submissions():
    return {
        'actions': ['python scripts/blend_submissions.py'],
        'file_dep': [
            'models/xgboost/submission_xgboost.csv',
            'models/catboost/submission_catboost.csv',
            'models/catboost/submission_catboost_bagged.csv'
        ],
        'targets': ['models/blended_submission.csv']
    }
