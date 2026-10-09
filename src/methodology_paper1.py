"""
Methodology 1: Adaptation of T. Talan (2026)
"Machine learning-based rainfall prediction across temporal scales:
model benchmarking and explainability analysis"

Key Pipeline Elements:
1. Missing-value handling
2. 3-standard-deviation outlier detection / clipping (winsorization)
3. Standardization (StandardScaler)
4. Temporal feature engineering (Lag 1, 2, 3; Rolling Mean 3d, 7d)
5. Feature relevance / selection
6. Tree-based ML models (Random Forest, Gradient Boosting, XGBoost, LightGBM) & MLP
7. SHAP explainability
"""
import time
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import SelectFromModel
try:
    import xgboost as xgb
except (ImportError, OSError):
    xgb = None

try:
    import lightgbm as lgb
except (ImportError, OSError):
    lgb = None

try:
    import shap
except (ImportError, OSError):
    shap = None

import matplotlib.pyplot as plt

from src.feature_engineering import create_lag_features, create_rolling_features, add_cyclic_calendar_features
from src.evaluation import compute_metrics


class TalanMethodologyAdaptation:
    def __init__(self, model_type='random_forest', random_state=42):
        self.name = "Paper 1 (Talan 2026 Adaptation)"
        self.paper_citation = (
            "T. Talan (2026). Machine learning-based rainfall prediction across temporal scales: "
            "model benchmarking and explainability analysis."
        )
        self.model_type = model_type
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.feature_selector = None
        self.selected_feature_names = []
        self.model = None
        self.feature_names_in = []
        self.metrics = {}
        self.shap_values = None
        self.explainer = None
        self.train_time = 0.0
        self.mean_3s_bounds = {}

    def _prepare_temporal_dataset(self, df):
        """Constructs temporal lag and rolling features chronologically without future leakage."""
        df_feat = df.copy()
        meteo_cols = [c for c in ['T2M', 'RH2M', 'PS', 'WS10M'] if c in df_feat.columns]
        
        # Cyclic calendar
        df_feat = add_cyclic_calendar_features(df_feat)
        
        # Lag features (1, 2, 3 days)
        df_feat, _ = create_lag_features(df_feat, meteo_cols, lags=[1, 2, 3])
        
        # Rolling features (3-day, 7-day past statistics)
        df_feat, _ = create_rolling_features(df_feat, meteo_cols, windows=[3, 7])
        
        return df_feat

    def fit_preprocess(self, df_train):
        """Fits 3-sigma bounds, scaler, and feature selector on training split ONLY."""
        df_prep = self._prepare_temporal_dataset(df_train)
        
        # Drop rows where initial lags are NaN
        df_prep = df_prep.dropna().reset_index(drop=True)
        
        y_train = df_prep['Rain'].values
        drop_cols = ['Rain', 'Date', 'YEAR', 'DOY', 'DISTRICT', 'District', 'STATE_UT_NAME', 'State']
        feature_cols = [c for c in df_prep.columns if c not in drop_cols and pd.api.types.is_numeric_dtype(df_prep[c])]
        self.feature_names_in = feature_cols
        
        X_train_raw = df_prep[feature_cols].copy()
        
        # 1. 3-Standard Deviation Outlier Clipping on Train
        X_train_clipped = X_train_raw.copy()
        for col in feature_cols:
            mean = X_train_raw[col].mean()
            std = X_train_raw[col].std()
            if std > 0:
                lower = mean - 3.0 * std
                upper = mean + 3.0 * std
                self.mean_3s_bounds[col] = (lower, upper)
                X_train_clipped[col] = np.clip(X_train_raw[col], lower, upper)
            else:
                self.mean_3s_bounds[col] = (-np.inf, np.inf)
                
        # 2. Standardization
        X_scaled = self.scaler.fit_transform(X_train_clipped)
        
        # 3. Feature relevance/selection via Tree Importance
        selector_rf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=self.random_state)
        selector_rf.fit(X_scaled, y_train)
        self.feature_selector = SelectFromModel(selector_rf, threshold='median', prefit=True)
        
        support = self.feature_selector.get_support()
        self.selected_feature_names = [f for f, s in zip(feature_cols, support) if s]
        
        X_selected = self.feature_selector.transform(X_scaled)
        return X_selected, y_train

    def transform_preprocess(self, df_test):
        """Transforms test set using fitted 3-sigma bounds, scaler, and feature selector."""
        df_prep = self._prepare_temporal_dataset(df_test)
        df_prep = df_prep.dropna().reset_index(drop=True)
        
        y_test = df_prep['Rain'].values if 'Rain' in df_prep.columns else None
        
        X_test_raw = df_prep[self.feature_names_in].copy()
        X_test_clipped = X_test_raw.copy()
        for col in self.feature_names_in:
            lower, upper = self.mean_3s_bounds.get(col, (-np.inf, np.inf))
            X_test_clipped[col] = np.clip(X_test_raw[col], lower, upper)
            
        X_scaled = self.scaler.transform(X_test_clipped)
        X_selected = self.feature_selector.transform(X_scaled)
        return X_selected, y_test, df_prep

    def fit(self, df_train):
        """Fits the specified model on the preprocessed training set."""
        start_time = time.time()
        X_train, y_train = self.fit_preprocess(df_train)
        
        if self.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=200, max_depth=8, min_samples_split=5,
                class_weight='balanced', random_state=self.random_state, n_jobs=-1
            )
        elif self.model_type == 'gradient_boosting':
            self.model = GradientBoostingClassifier(
                n_estimators=150, learning_rate=0.08, max_depth=5,
                random_state=self.random_state
            )
        elif self.model_type == 'xgboost':
            if xgb is not None:
                self.model = xgb.XGBClassifier(
                    n_estimators=150, max_depth=5, learning_rate=0.08,
                    eval_metric='logloss', random_state=self.random_state
                )
            else:
                self.model = GradientBoostingClassifier(n_estimators=150, learning_rate=0.08, max_depth=5, random_state=self.random_state)
        elif self.model_type == 'lightgbm':
            if lgb is not None:
                self.model = lgb.LGBMClassifier(
                    n_estimators=150, max_depth=5, learning_rate=0.08,
                    random_state=self.random_state, verbose=-1
                )
            else:
                self.model = GradientBoostingClassifier(n_estimators=150, learning_rate=0.08, max_depth=5, random_state=self.random_state)
        elif self.model_type == 'mlp':
            self.model = MLPClassifier(
                hidden_layer_sizes=(64, 32), max_iter=400,
                random_state=self.random_state, early_stopping=True
            )
        else:
            self.model = RandomForestClassifier(n_estimators=200, random_state=self.random_state)
            
        self.model.fit(X_train, y_train)
        self.train_time = time.time() - start_time
        return self

    def predict(self, df_test):
        X_test, _, _ = self.transform_preprocess(df_test)
        return self.model.predict(X_test)

    def predict_proba(self, df_test):
        X_test, _, _ = self.transform_preprocess(df_test)
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X_test)[:, 1]
        return None

    def evaluate(self, df_test):
        X_test, y_test, _ = self.transform_preprocess(df_test)
        y_pred = self.model.predict(X_test)
        y_prob = self.model.predict_proba(X_test)[:, 1] if hasattr(self.model, 'predict_proba') else None
        self.metrics = compute_metrics(y_test, y_pred, y_prob, self.train_time)
        return self.metrics

    def compute_shap_analysis(self, df_test, save_path=None):
        """Computes SHAP feature importance for Tree-based models and creates figure."""
        if not hasattr(self.model, 'feature_importances_'):
            return None
        X_test, _, _ = self.transform_preprocess(df_test)
        
        # Subsample for speed if needed
        sample_size = min(300, X_test.shape[0])
        X_sample = X_test[:sample_size]
        
        try:
            self.explainer = shap.TreeExplainer(self.model)
            shap_vals = self.explainer.shap_values(X_sample)
            
            # Format shap_values correctly depending on whether it's list (per class) or ndarray
            if isinstance(shap_vals, list):
                self.shap_values = shap_vals[1] if len(shap_vals) > 1 else shap_vals[0]
            elif isinstance(shap_vals, np.ndarray) and len(shap_vals.shape) == 3:
                self.shap_values = shap_vals[:, :, 1]
            else:
                self.shap_values = shap_vals
                
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.figure(figsize=(10, 6))
                shap.summary_plot(
                    self.shap_values, X_sample,
                    feature_names=self.selected_feature_names,
                    show=False
                )
                plt.title(f"{self.name} - SHAP Feature Explainability Summary", fontsize=12)
                plt.tight_layout()
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
                
            return self.shap_values
        except Exception as e:
            print(f"SHAP explanation computation warning: {e}")
            return None
