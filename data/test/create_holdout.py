"""
Generates an Unseen Holdout Dataset Scenario for Recommendation Testing
Extracts a strictly held-out chronological timeline (Year 2026) and formats it as an
unseen input dataset to test the profiling, recommendation, and prediction pipeline.
"""
import os
import pandas as pd
import numpy as np

project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
raw_data_path = os.path.join(project_dir, "data", "raw", "POWER_Point_Daily_Hyderabad.csv")

# Load raw dataset skipping NASA header
df = pd.read_csv(raw_data_path, skiprows=14)
df = df.replace(-999.0, np.nan).dropna(subset=['PRECTOTCORR']).reset_index(drop=True)

# Select unseen period: year 2026
df_holdout = df[df['YEAR'] == 2026].copy().reset_index(drop=True)

# Save to test directory
test_dir = os.path.join(project_dir, "data", "test")
os.makedirs(test_dir, exist_ok=True)
out_path = os.path.join(test_dir, "unseen_holdout_dataset.csv")
df_holdout.to_csv(out_path, index=False)
print(f"Created unseen holdout test dataset: {out_path} (N = {len(df_holdout)} days)")
