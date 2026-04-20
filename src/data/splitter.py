import pandas as pd
from src.config.config import TRAIN_RATIO, VAL_RATIO, TEST_RATIO

#split数据为时间序列格式的train, val, test
def time_series_split(df, date_col='date_GMT', train_ratio=None, val_ratio=None, test_ratio=None):
    if train_ratio is None:
        train_ratio = TRAIN_RATIO
    if val_ratio is None:
        val_ratio = VAL_RATIO
    if test_ratio is None:
        test_ratio = TEST_RATIO
    
    df_sorted = df.sort_values(date_col).reset_index(drop=True)
    n = len(df_sorted)
    
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train_df = df_sorted.iloc[:train_end].copy()
    val_df = df_sorted.iloc[train_end:val_end].copy()
    test_df = df_sorted.iloc[val_end:].copy()
    
    return train_df, val_df, test_df

#split数据为时间序列格式的cross validation
def time_series_cv_split(df, date_col='date_GMT', n_splits=5):
    df_sorted = df.sort_values(date_col).reset_index(drop=True)
    n = len(df_sorted)
    fold_size = n // (n_splits + 1)
    
    splits = []
    for i in range(n_splits):
        train_end = fold_size * (i + 1)
        val_start = train_end
        val_end = min(train_end + fold_size, n)
        
        train_df = df_sorted.iloc[:train_end].copy()
        val_df = df_sorted.iloc[val_start:val_end].copy()
        splits.append((train_df, val_df))
    
    return splits
