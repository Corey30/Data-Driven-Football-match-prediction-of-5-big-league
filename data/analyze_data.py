import pandas as pd

df = pd.read_csv('data/laliga_halftime_table.csv')
df['date_GMT'] = pd.to_datetime(df['date_GMT'], format='%b %d %Y - %I:%M%p', errors='coerce')
df = df.sort_values('date_GMT').reset_index(drop=True)

n = len(df)
train_end = int(n * 0.7)
val_end = int(n * 0.85)

print('=== Data Time Range ===')
print(f'Total records: {n}')
print(f'Training set: records 0-{train_end}, time: {df.iloc[0]["date_GMT"]} ~ {df.iloc[train_end-1]["date_GMT"]}')
print(f'Validation set: records {train_end}-{val_end}, time: {df.iloc[train_end]["date_GMT"]} ~ {df.iloc[val_end-1]["date_GMT"]}')
print(f'Test set: records {val_end}-{n}, time: {df.iloc[val_end]["date_GMT"]} ~ {df.iloc[n-1]["date_GMT"]}')

print('\n=== Season Distribution by Dataset ===')
print('Training set seasons:', df.iloc[:train_end]['season'].value_counts().to_dict())
print('Validation set seasons:', df.iloc[train_end:val_end]['season'].value_counts().to_dict())
print('Test set seasons:', df.iloc[val_end:]['season'].value_counts().to_dict())

print('\n=== Class Distribution by Dataset ===')
print('Training set:', df.iloc[:train_end]['target_match_result'].value_counts().to_dict())
print('Validation set:', df.iloc[train_end:val_end]['target_match_result'].value_counts().to_dict())
print('Test set:', df.iloc[val_end:]['target_match_result'].value_counts().to_dict())

print('\n=== Class Distribution by Season ===')
for season in df['season'].unique():
    season_df = df[df['season'] == season]
    dist = season_df['target_match_result'].value_counts(normalize=True).to_dict()
    print(f'{season}: H={dist.get("H",0):.2%}, D={dist.get("D",0):.2%}, A={dist.get("A",0):.2%}, count={len(season_df)}')
