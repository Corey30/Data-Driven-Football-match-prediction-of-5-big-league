import pandas as pd
import numpy as np
from src.config.config import (
    HALFTIME_COLUMNS, IDENTIFIER_COLUMNS, TARGET_COLUMNS, TARGET_COLUMN,
    LABEL_ENCODING
)
from src.features.prematch_features import (
    handle_missing_values, get_feature_names
)


def get_halftime_feature_columns(df):
    exclude_cols = set(IDENTIFIER_COLUMNS + TARGET_COLUMNS)
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    return feature_cols


def build_halftime_features(df):
    df = df.copy()
    feature_cols = get_halftime_feature_columns(df)
    X = df[feature_cols].copy()
    y = df[TARGET_COLUMN].copy() if TARGET_COLUMN in df.columns else None
    return X, y


def encode_ht_result(X):
    X = X.copy()
    if 'ht_result' in X.columns:
        X['ht_result_encoded'] = X['ht_result'].map(LABEL_ENCODING)
        X = X.drop(columns=['ht_result'])
    return X


def process_halftime_features(df):
    X, y = build_halftime_features(df)
    X = encode_ht_result(X)
    X = handle_missing_values(X)
    return X, y
