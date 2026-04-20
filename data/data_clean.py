import pandas as pd
from pathlib import Path

df=pd.read_csv('laliga_halftime_table.csv')
cols_to_drop=[]
df.drop(columns=[c for c in cols_to_drop if c in df.columns], inplace=True)

df.to_csv('laliga_halftime_table_clean.csv', index=False)