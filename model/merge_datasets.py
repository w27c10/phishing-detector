"""Merges all dataset_*.csv files in the current directory into dataset_combined.csv."""
import glob
import pandas as pd

files = glob.glob('dataset*.csv')
print(f'Found: {files}')

dfs = []
for f in files:
    if 'combined' in f or 'features' in f:
        continue
    df = pd.read_csv(f, encoding='utf-8', on_bad_lines='skip')
    # Keep only the three columns we need; ignore any extra feature columns
    for col in ('url', 'dom', 'label'):
        if col not in df.columns:
            df[col] = '' if col != 'label' else 0
    df = df[['url', 'dom', 'label']].copy()
    df['url'] = df['url'].fillna('').astype(str).str.replace('\x00', '', regex=False)
    df['dom'] = df['dom'].fillna('').astype(str).str.replace('\x00', '', regex=False)
    df['label'] = pd.to_numeric(df['label'], errors='coerce').fillna(0).astype(int)
    df = df[df['url'].str.startswith('http')]
    dfs.append(df)

combined = pd.concat(dfs).drop_duplicates('url').sample(frac=1, random_state=42).reset_index(drop=True)

n_phish = int(combined['label'].sum())
n_safe  = len(combined) - n_phish
print(f'Combined: {len(combined)} rows  |  Phishing: {n_phish}  |  Safe: {n_safe}')

combined.to_pickle('dataset_combined.pkl')
print('Saved → dataset_combined.pkl')
