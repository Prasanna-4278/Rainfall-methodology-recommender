"""
Dataset Profiling & Meta-Feature Extraction Module
Extracts statistical, distributional, class balance, and temporal characteristics
to create a structured dataset profile and meta-feature vector for methodology recommendation.
GUARANTEES 100% JSON-compliant meta-features (zero NaNs).
"""
import pandas as pd
import numpy as np
from scipy import stats


def profile_dataset(df, target_col='Rain', date_col='Date', raw_rainfall_series=None):
    """
    Computes a comprehensive profile of any meteorological or rainfall dataset.
    Safely handles unlabeled datasets, cross-sectional records, and varied column names.
    """
    n_samples = int(len(df))
    
    # Find all numeric features
    numeric_df = df.select_dtypes(include=[np.number])
    exclude_cols = [c for c in ['Rain', 'Date', 'YEAR', 'DOY', target_col] if c is not None]
    feature_cols = [c for c in numeric_df.columns if c not in exclude_cols]
    n_features = int(len(feature_cols))
    
    if n_samples == 0:
        return {
            'n_samples': 0, 'n_features': n_features, 'feature_names': feature_cols,
            'missing_pct': 0.0, 'outlier_pct_3sigma': 0.0, 'outlier_pct_iqr': 0.0,
            'rain_pct': None, 'no_rain_pct': None, 'imbalance_ratio': None,
            'mean_skewness': 0.0, 'max_skewness': 0.0, 'avg_feature_corr': 0.0, 'max_feature_corr': 0.0,
            'has_temporal_order': False, 'temporal_autocorr_lag1': 0.0, 'seasonality_strength': 0.0,
            'rainfall_variability': None, 'sample_to_feature_ratio': 0.0
        }, "Dataset contains 0 rows."

    # 1. Missingness
    total_cells = max(1, n_samples * n_features)
    missing_cells = int(df[feature_cols].isna().sum().sum()) if n_features > 0 else 0
    missing_pct = float(round(100.0 * missing_cells / total_cells, 2))
    
    # 2. Outliers (using 3-sigma and IQR)
    outlier_count_3s = 0
    outlier_count_iqr = 0
    for col in feature_cols:
        vals = df[col].dropna()
        if len(vals) > 5 and vals.std() > 0:
            # 3-sigma
            z = np.abs((vals - vals.mean()) / vals.std())
            outlier_count_3s += int((z > 3).sum())
            # IQR
            q25, q75 = vals.quantile(0.25), vals.quantile(0.75)
            iqr = q75 - q25
            outlier_count_iqr += int(((vals < (q25 - 1.5 * iqr)) | (vals > (q75 + 1.5 * iqr))).sum())
            
    outlier_pct_3s = float(round(100.0 * outlier_count_3s / total_cells, 2))
    outlier_pct_iqr = float(round(100.0 * outlier_count_iqr / total_cells, 2))
    
    # 3. Class balance (if target exists)
    has_target = target_col is not None and target_col in df.columns and len(df[target_col].dropna()) > 0
    if has_target:
        rain_count = int((df[target_col] == 1).sum())
        no_rain_count = int((df[target_col] == 0).sum())
        rain_pct = float(round(100.0 * rain_count / n_samples, 2))
        no_rain_pct = float(round(100.0 * no_rain_count / n_samples, 2))
        min_c = min(rain_count, no_rain_count)
        max_c = max(rain_count, no_rain_count)
        imbalance_ratio = float(round(max_c / min_c if min_c > 0 else 999.0, 2))
    else:
        rain_pct, no_rain_pct, imbalance_ratio = None, None, None
        
    # 4. Skewness and correlations
    if n_features > 0:
        skew_series = df[feature_cols].skew().dropna()
        skew_vals = np.abs(skew_series.values)
        mean_skewness = float(round(float(np.mean(skew_vals)), 3)) if len(skew_vals) > 0 else 0.0
        max_skewness = float(round(float(np.max(skew_vals)), 3)) if len(skew_vals) > 0 else 0.0
    else:
        mean_skewness, max_skewness = 0.0, 0.0
        
    avg_feature_corr, max_feature_corr = 0.0, 0.0
    if n_features > 1:
        corr_matrix = df[feature_cols].corr().abs()
        corr_vals = corr_matrix.to_numpy(copy=True)
        np.fill_diagonal(corr_vals, np.nan)
        valid_corrs = corr_vals[~np.isnan(corr_vals)]
        if len(valid_corrs) > 0:
            avg_feature_corr = float(round(float(np.mean(valid_corrs)), 3))
            max_feature_corr = float(round(float(np.max(valid_corrs)), 3))
            
    # 5. Temporal characteristics
    has_temporal = False
    if date_col in df.columns and df[date_col].notna().any():
        has_temporal = True
    elif 'YEAR' in df.columns and 'DOY' in df.columns:
        has_temporal = True
        
    mean_autocorr = 0.0
    if has_temporal and n_features > 0 and n_samples > 10:
        autocorr_list = []
        for col in feature_cols:
            s = df[col].dropna()
            if len(s) > 10:
                try:
                    ac1 = s.autocorr(lag=1)
                    if np.isfinite(ac1):
                        autocorr_list.append(abs(ac1))
                except Exception:
                    pass
        if len(autocorr_list) > 0:
            mean_autocorr = float(round(float(np.mean(autocorr_list)), 3))
            
    # Seasonality check
    seasonality_strength = 0.0
    if has_temporal and date_col in df.columns and df[date_col].notna().any():
        try:
            months = df[date_col].dt.month
            main_col = feature_cols[0] if len(feature_cols) > 0 else None
            if main_col:
                groups = [g.dropna().values for _, g in df.groupby(months)[main_col] if len(g.dropna()) > 1]
                if len(groups) > 1:
                    f_stat, _ = stats.f_oneway(*groups)
                    if np.isfinite(f_stat):
                        seasonality_strength = float(round(min(float(f_stat) / 100.0, 1.0), 3))
        except Exception:
            seasonality_strength = float(round(mean_autocorr * 0.8, 3))
    elif has_temporal:
        seasonality_strength = float(round(mean_autocorr * 0.8, 3))
        
    # Rainfall variability
    cv_rain = None
    if raw_rainfall_series is not None and len(raw_rainfall_series.dropna()) > 0:
        rf = raw_rainfall_series.dropna()
        if rf.mean() > 0:
            cv_val = float(rf.std() / rf.mean())
            if np.isfinite(cv_val):
                cv_rain = float(round(cv_val, 3))
    elif has_target and df[target_col].mean() > 0:
        cv_val = float(df[target_col].std() / df[target_col].mean())
        if np.isfinite(cv_val):
            cv_rain = float(round(cv_val, 3))
            
    sample_to_feat = float(round(n_samples / max(n_features, 1), 1))
    
    profile = {
        'n_samples': n_samples,
        'n_features': n_features,
        'feature_names': feature_cols,
        'missing_pct': missing_pct,
        'outlier_pct_3sigma': outlier_pct_3s,
        'outlier_pct_iqr': outlier_pct_iqr,
        'rain_pct': rain_pct,
        'no_rain_pct': no_rain_pct,
        'imbalance_ratio': imbalance_ratio,
        'mean_skewness': mean_skewness,
        'max_skewness': max_skewness,
        'avg_feature_corr': avg_feature_corr,
        'max_feature_corr': max_feature_corr,
        'has_temporal_order': has_temporal,
        'temporal_autocorr_lag1': mean_autocorr,
        'seasonality_strength': seasonality_strength,
        'rainfall_variability': cv_rain,
        'sample_to_feature_ratio': sample_to_feat
    }
    
    profile_text = f"""==================================================
DATASET PROFILE & CHARACTERISTICS
==================================================
Samples (N)             : {profile['n_samples']}
Features (D)            : {profile['n_features']} ({', '.join(feature_cols[:8])}{'...' if len(feature_cols) > 8 else ''})
Sample-to-Feature Ratio : {profile['sample_to_feature_ratio']}
Missingness             : {profile['missing_pct']}%
Outliers (3-sigma)      : {profile['outlier_pct_3sigma']}%
Outliers (IQR)          : {profile['outlier_pct_iqr']}%
Rain Events             : {str(profile['rain_pct']) + '%' if profile['rain_pct'] is not None else 'Unlabeled'}
No-Rain Events          : {str(profile['no_rain_pct']) + '%' if profile['no_rain_pct'] is not None else 'Unlabeled'}
Class Imbalance Ratio   : {str(profile['imbalance_ratio']) + ':1' if profile['imbalance_ratio'] is not None else 'N/A'}
Mean Feature Skewness   : {profile['mean_skewness']}
Average Pairwise Corr   : {profile['avg_feature_corr']}
Max Pairwise Corr       : {profile['max_feature_corr']}
Temporal Continuity     : {'Yes' if profile['has_temporal_order'] else 'No'}
Autocorrelation (Lag 1) : {profile['temporal_autocorr_lag1']}
Seasonality Strength    : {profile['seasonality_strength']}
Rainfall Variability CV : {profile['rainfall_variability'] if profile['rainfall_variability'] is not None else 'N/A'}
=================================================="""
    
    return profile, profile_text
