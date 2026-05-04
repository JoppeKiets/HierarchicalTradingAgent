import numpy as np
import os

# Sample ticker
sample_ticker = 'HEWJ'

# Paths
base = 'data/feature_cache'
daily_dir = os.path.join(base, 'daily')
minute_news_dir = os.path.join(base, 'minute_news')

def print_npy_info(path):
    try:
        arr = np.load(path, allow_pickle=True)
        print(f"{os.path.basename(path)}: shape={getattr(arr, 'shape', None)}, dtype={getattr(arr, 'dtype', None)}")
        if arr.size < 10:
            print(f"  values: {arr}")
        else:
            print(f"  sample: {arr[:5]}")
    except Exception as e:
        print(f"Failed to load {path}: {e}")

# Daily data
print('--- DAILY DATA ---')
for suffix in ['_dates.npy', '_features.npy', '_targets.npy']:
    print_npy_info(os.path.join(daily_dir, f'{sample_ticker}{suffix}'))

# Minute news data
print('\n--- MINUTE NEWS DATA ---')
for suffix in ['_news_features.npy', '_news_timestamps.npy']:
    print_npy_info(os.path.join(minute_news_dir, f'{sample_ticker}{suffix}'))
