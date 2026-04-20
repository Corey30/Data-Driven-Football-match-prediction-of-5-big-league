import pandas as pd
from src.config.config import RAW_DATA_PATH


def load_data(filepath=None):
    if filepath is None:
        filepath = RAW_DATA_PATH
    
    encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
    for encoding in encodings:
        try:
            df = pd.read_csv(filepath, encoding=encoding)
            return df
        except UnicodeDecodeError:
            continue
    
    raise ValueError(f"Cannot read file {filepath} with any supported encoding")


def preprocess_data(df):#sort by time
    df = df.copy()
    if 'date_GMT' in df.columns:
        df['date_GMT'] = pd.to_datetime(df['date_GMT'], format='%b %d %Y - %I:%M%p', errors='coerce')
    df = df.sort_values('date_GMT').reset_index(drop=True)
    return df


EXCLUDE_SEASONS = ['2025-2026']

def clean_data(df):
    df = df.copy()
    df = df.dropna(subset=['target_match_result'])
    if 'season' in df.columns:
        df = df[~df['season'].isin(EXCLUDE_SEASONS)]
    return df
