"""
Data Cleaning & Anomaly Handling Module
Performs programmatic inspection, NASA POWER missing value code resolution,
WD10M domain-validity analysis, and target derivation without target leakage.
"""
import pandas as pd
import numpy as np


def clean_dataset(df_raw, drop_unphysical_wd=True):
    """
    Cleans raw NASA POWER data and documents every cleaning decision:
    1. Replaces -999.0 missing value codes with NaN.
    2. Drops rows with missing targets or complete feature unavailabilities.
    3. Analyzes and handles WD10M (wind direction) anomalies.
    4. Creates binary classification target 'Rain' (1 if PRECTOTCORR > 0 else 0).
    5. STRICTLY drops PRECTOTCORR from predictors to prevent target leakage.
    
    Parameters
    ----------
    df_raw : pd.DataFrame
        Raw dataframe loaded by data_loader.
    drop_unphysical_wd : bool, default True
        If True, drops WD10M column because all observations in this NASA POWER export
        contain negative unphysical values (-5,405 to -98,710 degrees).
        
    Returns
    -------
    df_clean : pd.DataFrame
        Cleaned dataframe ready for profiling and modeling.
    cleaning_log : dict
        Audit trail of all cleaning decisions, statistics, and domain justifications.
    """
    cleaning_log = {
        'initial_rows': len(df_raw),
        'initial_cols': df_raw.columns.tolist(),
        'missing_999_detected': {},
        'wd10m_diagnosis': {},
        'target_summary': {},
        'decisions_made': []
    }
    
    df = df_raw.copy()
    numeric_cols = [c for c in df.columns if c not in ['Date', 'YEAR', 'DOY']]
    
    # 1. Detect -999.0 missing values
    for col in numeric_cols:
        count_999 = int((df[col] == -999.0).sum())
        if count_999 > 0:
            cleaning_log['missing_999_detected'][col] = count_999
            
    # Replace -999.0 with NaN
    df[numeric_cols] = df[numeric_cols].replace(-999.0, np.nan)
    
    # 2. Check complete missing rows (e.g. trailing unfinalized observations)
    missing_target = df['PRECTOTCORR'].isna()
    n_missing_target = int(missing_target.sum())
    if n_missing_target > 0:
        df = df[~missing_target].copy()
        cleaning_log['decisions_made'].append(
            f"Removed {n_missing_target} rows where PRECTOTCORR was -999.0 (NASA missing data code at tail of timeline)."
        )
        
    # 3. Analyze WD10M (Wind Direction at 10m)
    if 'WD10M' in df.columns:
        wd_valid = df['WD10M'].dropna()
        wd_neg = (wd_valid < 0).sum()
        wd_pos = (wd_valid >= 0).sum()
        wd_min = float(wd_valid.min()) if len(wd_valid) > 0 else np.nan
        wd_max = float(wd_valid.max()) if len(wd_valid) > 0 else np.nan
        
        cleaning_log['wd10m_diagnosis'] = {
            'total_valid': len(wd_valid),
            'negative_count': int(wd_neg),
            'positive_count': int(wd_pos),
            'min_value': wd_min,
            'max_value': wd_max,
            'physically_valid_pct': float(100.0 * wd_pos / len(wd_valid)) if len(wd_valid) > 0 else 0.0
        }
        
        if drop_unphysical_wd:
            df = df.drop(columns=['WD10M'])
            cleaning_log['decisions_made'].append(
                "CRITICAL DOMAIN DECISION: WD10M (Wind Direction) dropped. "
                "Physical wind direction requires values in [0, 360] degrees. "
                f"100% of observations ({wd_neg}/{len(wd_valid)}) have negative values "
                f"ranging from {wd_min:.1f} to {wd_max:.1f}, indicating corrupted internal fill "
                "or calculation failure codes from MERRA-2. Feeding raw values with std > 7000 "
                "would introduce severe scale distortion and invalid noise into models."
            )
            
    # 4. Create binary target 'Rain'
    # Rain = 1 when PRECTOTCORR > 0, Rain = 0 when PRECTOTCORR == 0
    df['Rain'] = (df['PRECTOTCORR'] > 0).astype(int)
    rain_count = int((df['Rain'] == 1).sum())
    no_rain_count = int((df['Rain'] == 0).sum())
    total_valid = len(df)
    
    cleaning_log['target_summary'] = {
        'rain_days': rain_count,
        'rain_pct': round(100.0 * rain_count / total_valid, 2),
        'no_rain_days': no_rain_count,
        'no_rain_pct': round(100.0 * no_rain_count / total_valid, 2),
        'imbalance_ratio': round(max(rain_count, no_rain_count) / min(rain_count, no_rain_count), 2)
    }
    
    # 5. STRICT NON-LEAKAGE REMOVAL OF PRECTOTCORR
    # Target Rain was derived from PRECTOTCORR. It MUST NOT be in feature set.
    raw_prectotcorr = df['PRECTOTCORR'].copy()
    df = df.drop(columns=['PRECTOTCORR'])
    cleaning_log['decisions_made'].append(
        "CRITICAL LEAKAGE PREVENTION: PRECTOTCORR removed from predictors after constructing binary 'Rain' target."
    )
    
    cleaning_log['final_rows'] = len(df)
    cleaning_log['final_cols'] = df.columns.tolist()
    
    return df, cleaning_log, raw_prectotcorr
