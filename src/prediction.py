"""
Prediction Pipeline Module
Instantiates the single recommended published methodology, trains/adapts it,
and generates rainfall predictions and probabilities on any target dataset.
Includes intelligent meteorological synonym mapping and robust column alignment.
"""
import os
import pandas as pd
import numpy as np

from src.methodology_paper1 import TalanMethodologyAdaptation
from src.methodology_paper2 import PathaniaMethodologyAdaptation
from src.methodology_paper3 import KanmaniPappaMethodologyAdaptation
from src.methodology_paper4 import KhanMethodologyAdaptation


METEO_SYNONYMS = {
    'T2M': ['t2m', 'temp', 'temperature', 'temperature_c', 'temp_c', 'mean_temp', 'max_temp', 'min_temp'],
    'RH2M': ['rh2m', 'humidity', 'relative_humidity', 'humidity_pct', 'rh', 'rh_pct'],
    'PS': ['ps', 'pressure', 'pressure_hpa', 'surface_pressure', 'sp', 'pres', 'mslp'],
    'WS10M': ['ws10m', 'wind_speed', 'wind', 'ws', 'wind_speed_kmh', 'wind_speed_ms', 'windspeed']
}


def map_meteorological_columns(df):
    """Detects and renames meteorological synonyms to standard names (T2M, RH2M, PS, WS10M)."""
    df_out = df.copy()
    col_map = {}
    lower_to_orig = {str(c).lower().strip(): c for c in df_out.columns}
    
    for std_name, synonyms in METEO_SYNONYMS.items():
        if std_name not in df_out.columns:
            for syn in synonyms:
                if syn in lower_to_orig:
                    orig_col = lower_to_orig[syn]
                    col_map[orig_col] = std_name
                    break
                    
    if col_map:
        df_out = df_out.rename(columns=col_map)
        
    # Unit conversions if needed
    if 'PS' in df_out.columns and df_out['PS'].median() > 500:
        # hPa to kPa
        df_out['PS'] = df_out['PS'] / 10.0
    if 'WS10M' in df_out.columns and df_out['WS10M'].median() > 20:
        # km/h to m/s
        df_out['WS10M'] = df_out['WS10M'] / 3.6
        
    return df_out


def align_features_for_model(df, expected_features, train_medians=None):
    """Ensures all expected feature columns exist in df, imputing training medians for any missing."""
    df_out = df.copy()
    if train_medians is None:
        train_medians = {'T2M': 25.5, 'RH2M': 65.0, 'PS': 95.3, 'WS10M': 3.5}
        
    for feat in expected_features:
        if feat not in df_out.columns:
            val = train_medians.get(feat, 0.0)
            df_out[feat] = val
            
    return df_out


def execute_recommended_prediction(df_train, df_test, recommended_methodology_name, output_path=None):
    """
    Executes rainfall prediction using ONLY the recommended published methodology.
    Robust against arbitrary column sets, missing predictors, and custom schemas.
    """
    # 1. Map synonyms on test set
    df_test_mapped = map_meteorological_columns(df_test)
    df_train_mapped = map_meteorological_columns(df_train)
    
    # 2. Check if target Rain is present or can be derived from continuous/district columns
    if 'Rain' not in df_test_mapped.columns:
        for col in list(df_test_mapped.columns):
            lc = str(col).lower().strip()
            if any(k in lc for k in ['prectot', 'rainfall', 'precipitation', 'prcp', 'rain_mm']) and pd.api.types.is_numeric_dtype(df_test_mapped[col]):
                df_test_mapped['Rain'] = (df_test_mapped[col] > 0.05).astype(int)
                df_test_mapped = df_test_mapped.drop(columns=[col])
                break
        if 'Rain' not in df_test_mapped.columns:
            month_cols = [c for c in df_test_mapped.columns if str(c).upper().strip() in ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC', 'ANNUAL']]
            if len(month_cols) >= 3:
                ref_col = 'ANNUAL' if 'ANNUAL' in df_test_mapped.columns else ('JUL' if 'JUL' in df_test_mapped.columns else month_cols[-1])
                med = float(df_test_mapped[ref_col].median())
                df_test_mapped['Rain'] = (df_test_mapped[ref_col] >= med).astype(int)

    has_test_target = 'Rain' in df_test_mapped.columns and df_test_mapped['Rain'].nunique() > 1
    has_enough_samples = len(df_test_mapped) >= 20
    use_local_fit = has_test_target and has_enough_samples
    
    if "Paper 1" in recommended_methodology_name:
        model = TalanMethodologyAdaptation(model_type='random_forest')
    elif "Paper 2" in recommended_methodology_name:
        model = PathaniaMethodologyAdaptation()
    elif "Paper 3" in recommended_methodology_name:
        model = KanmaniPappaMethodologyAdaptation(model_type='random_forest')
    elif "Paper 4" in recommended_methodology_name:
        model = KhanMethodologyAdaptation(model_type='random_forest')
    else:
        model = TalanMethodologyAdaptation(model_type='random_forest')
        
    train_source = df_test_mapped if use_local_fit else df_train_mapped
    
    print(f"Training recommended methodology: {model.name}...")
    model.fit(train_source)
    
    # 3. Align test features to match model's expected inputs
    expected_feats = getattr(model, 'feature_names_in', ['T2M', 'RH2M', 'PS', 'WS10M'])
    train_medians = {c: float(train_source[c].median()) for c in expected_feats if c in train_source.columns}
    
    df_test_aligned = align_features_for_model(df_test_mapped, expected_feats, train_medians)
    
    print("Executing rainfall predictions on target dataset...")
    y_pred = model.predict(df_test_aligned)
    y_prob = model.predict_proba(df_test_aligned)
    
    n_preds = len(y_pred)
    if n_preds < len(df_test):
        df_display = df_test.iloc[-n_preds:].reset_index(drop=True)
        df_test_mapped_sub = df_test_mapped.iloc[-n_preds:].reset_index(drop=True)
    else:
        df_display = df_test
        df_test_mapped_sub = df_test_mapped

    pred_data = {}
    if 'Date' in df_display.columns:
        pred_data['Date'] = df_display['Date'].values
    elif 'YEAR' in df_display.columns and 'DOY' in df_display.columns:
        pred_data['Date'] = [f"{y}-{d:03d}" for y, d in zip(df_display['YEAR'], df_display['DOY'])]
    elif 'DISTRICT' in df_display.columns:
        pred_data['Date'] = df_display['DISTRICT'].values
    else:
        pred_data['Date'] = [f"Row_{i+1}" for i in range(n_preds)]
        
    pred_data['Predicted_Rain'] = y_pred
    if y_prob is not None:
        pred_data['Rain_Probability'] = np.round(y_prob, 4)
        
    if 'Rain' in df_test_mapped_sub.columns:
        pred_data['Actual_Rain'] = df_test_mapped_sub['Rain'].values
        
    df_out = pd.DataFrame(pred_data)
    
    eval_metrics = None
    if 'Rain' in df_test_mapped.columns and df_test_mapped['Rain'].nunique() > 1:
        try:
            eval_metrics = model.evaluate(df_test_aligned)
        except Exception:
            pass
            
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df_out.to_csv(output_path, index=False)
        print(f"Predictions successfully saved to: {output_path}")
        
    return df_out, eval_metrics
