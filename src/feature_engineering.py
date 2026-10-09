"""
Feature Engineering Module
Provides strict chronological temporal features, lag features,
rolling statistics, and cyclic trigonometric encodings without future leakage.
"""
import pandas as pd
import numpy as np


def add_cyclic_calendar_features(df, doy_col='DOY'):
    """
    Adds cyclical sine and cosine transformations of Day of Year (DOY).
    Captures annual astronomical solar periodicity.
    """
    df_out = df.copy()
    if doy_col in df_out.columns:
        doy_rad = 2 * np.pi * df_out[doy_col] / 365.25
        df_out['sin_doy'] = np.sin(doy_rad)
        df_out['cos_doy'] = np.cos(doy_rad)
    return df_out


def create_lag_features(df, feature_cols, lags=[1, 2, 3]):
    """
    Constructs past temporal lag features for specified meteorological variables.
    Strictly past-oriented: t - l where l >= 1 (ZERO future leakage).
    
    Parameters
    ----------
    df : pd.DataFrame
        Chronologically sorted dataframe.
    feature_cols : list
        List of continuous meteorological features to lag.
    lags : list of int
        Lag steps (e.g. [1, 2, 3] days).
        
    Returns
    -------
    df_out : pd.DataFrame
        Dataframe with added lag columns.
    new_cols : list
        Names of newly created lag columns.
    """
    df_out = df.copy()
    new_cols = []
    for col in feature_cols:
        for lag in lags:
            col_name = f"{col}_lag{lag}"
            df_out[col_name] = df_out[col].shift(lag)
            new_cols.append(col_name)
    return df_out, new_cols


def create_rolling_features(df, feature_cols, windows=[3, 7]):
    """
    Constructs rolling window mean and standard deviation features.
    Strictly applies shift(1) prior to rolling aggregation to ensure observation
    at time t NEVER includes itself in the rolling window.
    
    Parameters
    ----------
    df : pd.DataFrame
        Chronologically sorted dataframe.
    feature_cols : list
        List of continuous meteorological features.
    windows : list of int
        Rolling window sizes in days (e.g. [3, 7]).
        
    Returns
    -------
    df_out : pd.DataFrame
        Dataframe with added rolling columns.
    new_cols : list
        Names of newly created rolling columns.
    """
    df_out = df.copy()
    new_cols = []
    for col in feature_cols:
        for w in windows:
            mean_col = f"{col}_roll_mean_{w}d"
            std_col = f"{col}_roll_std_{w}d"
            # Shift 1 day into the past before rolling
            past_series = df_out[col].shift(1)
            df_out[mean_col] = past_series.rolling(window=w, min_periods=1).mean()
            df_out[std_col] = past_series.rolling(window=w, min_periods=1).std().fillna(0.0)
            new_cols.extend([mean_col, std_col])
    return df_out, new_cols
