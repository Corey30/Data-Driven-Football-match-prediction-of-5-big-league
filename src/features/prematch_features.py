import pandas as pd
import numpy as np
from src.config.config import (
    HALFTIME_COLUMNS, IDENTIFIER_COLUMNS, TARGET_COLUMNS, TARGET_COLUMN
)


def get_prematch_feature_columns(df):
    exclude_cols = set(HALFTIME_COLUMNS + IDENTIFIER_COLUMNS + TARGET_COLUMNS)
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    return feature_cols


def build_prematch_features(df):
    df = df.copy()
    feature_cols = get_prematch_feature_columns(df)
    X = df[feature_cols].copy()
    y = df[TARGET_COLUMN].copy() if TARGET_COLUMN in df.columns else None
    return X, y


def handle_missing_values(X, strategy='median'):
    X = X.copy()
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    X = X[numeric_cols]
    
    if strategy == 'median':
        X = X.fillna(X.median())
    elif strategy == 'mean':
        X = X.fillna(X.mean())
    elif strategy == 'zero':
        X = X.fillna(0)
    
    X = X.fillna(0)
    
    return X


def encode_labels(y):
    from src.config.config import LABEL_ENCODING
    return y.map(LABEL_ENCODING)


def decode_labels(y_encoded):
    from src.config.config import LABEL_DECODING
    if isinstance(y_encoded, pd.Series):
        return y_encoded.map(LABEL_DECODING)
    return pd.Series(y_encoded).map(LABEL_DECODING)


def get_feature_names(X):
    return X.columns.tolist()
