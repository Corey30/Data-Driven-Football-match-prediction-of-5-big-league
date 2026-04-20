import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent

DATA_DIR = BASE_DIR / 'data'
RAW_DATA_PATH = DATA_DIR / 'laliga_halftime_table.csv'
OUTPUT_DIR = BASE_DIR / 'outputs'
MODEL_DIR = OUTPUT_DIR / 'models'
REPORT_DIR = OUTPUT_DIR / 'reports'

HALFTIME_COLUMNS = [
    'home_team_goal_count_half_time',
    'away_team_goal_count_half_time',
    'total_goals_at_half_time',
    'ht_result',
    'home_team_first_half_cards',
    'away_team_first_half_cards',
    'half_time_lead'
]

IDENTIFIER_COLUMNS = [
    'match_id', 'timestamp', 'date_GMT', 'season', 'status',
    'stadium_name', 'referee', 'home_team_name', 'away_team_name',
    'Game Week'
]

TARGET_COLUMNS = [
    'target_match_result',
    'target_home_team_goal_count',
    'target_away_team_goal_count',
    'target_total_goal_count'
]

TARGET_COLUMN = 'target_match_result'

LABEL_ENCODING = {'H': 0, 'D': 1, 'A': 2}
LABEL_DECODING = {0: 'H', 1: 'D', 2: 'A'}

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

RANDOM_SEED = 42

MODEL_PARAMS = {
    'random_forest': {
        'n_estimators': 100,
        'max_depth': 4,
        'min_samples_split': 30,
        'min_samples_leaf': 15,
        'max_features': 'sqrt',
        'random_state': 42,
        'class_weight': 'balanced'
    },
    'logistic_regression': {
        'C': 0.01,
        'random_state': 42,
        'class_weight': 'balanced',
        'solver': 'lbfgs',
        'max_iter': 1000,
        'penalty': 'l2'             # 显式指定
    },
    'xgboost': {
        'n_estimators': 100,
        'max_depth': 2,
        'learning_rate': 0.03,
        'subsample': 0.6,
        'colsample_bytree': 0.5,
        'reg_lambda': 10.0,
        'reg_alpha': 2.0,
        'random_state': 42,
        'use_label_encoder': False,
        'eval_metric': 'mlogloss'
    },
    'naive_bayes': {
        'var_smoothing': 1e-06       # 稍微调大一点点，平滑更多
    },
    'svm': {
        'C': 0.1,                   # 比LR稍微松一点的正则化
        'kernel': 'linear',
        'probability': True,
        'random_state': 42,
        'class_weight': 'balanced'
    },
    'fnn': {
        'hidden_layer_sizes': (16,), # 极窄网络
        'activation': 'relu',
        'learning_rate_init': 0.001,
        'max_iter': 200,
        'random_state': 42,
        'early_stopping': True,
        'validation_fraction': 0.1,
        'alpha': 0.1
    }
}

GRID_SEARCH_PARAMS = {
    'random_forest': {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 8, 12],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'class_weight': [None, 'balanced']
    },
    'logistic_regression': {
        'C': [0.01, 0.1, 1, 3, 10],
        'class_weight': [None, 'balanced'],
        'solver': ['lbfgs']
    },
    'xgboost': {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.3]
    },
    'naive_bayes': {
        'var_smoothing': [1e-9, 1e-8, 1e-7]
    },
    'svm': {
        'C': [0.1, 1, 10],
        'kernel': ['rbf', 'linear'],
        'class_weight': [None, 'balanced']
    },
    'fnn': {
        'hidden_layer_sizes': [(64,), (128, 64), (256, 128, 64)],
        'activation': ['relu', 'tanh'],
        'learning_rate_init': [0.001, 0.01]
    }
}

MODEL_DISPLAY_NAMES = {
    'random_forest': 'Random Forest',
    'logistic_regression': 'Logistic Regression',
    'xgboost': 'XGBoost',
    'naive_bayes': 'Naive Bayes',
    'svm': 'SVM',
    'fnn': 'Feedforward NN'
}
