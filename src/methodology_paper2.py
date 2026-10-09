"""
Methodology 2: Adaptation of Anuj Pathania et al. (2024)
"Harnessing Machine Learning for Precise Rainfall Forecasting: A Comparative Study"

Key Pipeline Elements:
1. Missing-value handling
2. Outlier handling via IQR capping
3. Domain feature engineering (seasonal quarters and meteorological bins)
4. Categorical / ordinal encoding
5. Logistic Regression with calibrated probability prediction
6. Comprehensive evaluation (Acc, Prec, Rec, F1, ROC-AUC, Confusion Matrix)
"""
import time
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.evaluation import compute_metrics


class PathaniaMethodologyAdaptation:
    def __init__(self, random_state=42):
        self.name = "Paper 2 (Pathania et al. 2024 Adaptation)"
        self.paper_citation = (
            "Anuj Pathania et al. (2024). Harnessing Machine Learning for Precise Rainfall Forecasting: "
            "A Comparative Study."
        )
        self.random_state = random_state
        self.iqr_bounds = {}
        self.scaler = StandardScaler()
        self.model = None
        self.feature_names_in = []
        self.metrics = {}
        self.train_time = 0.0

    def _engineer_features(self, df):
        """Engineers seasonal bins and domain features as in Pathania et al."""
        df_out = df.copy()
        
        # Seasonal quarter categorical indicator
        if 'DOY' in df_out.columns:
            # 1: Winter (DOY 1-60), 2: Pre-monsoon/Summer (61-150), 3: Monsoon (151-275), 4: Post-monsoon (276-366)
            bins = [0, 60, 150, 275, 367]
            labels = [1, 2, 3, 4]
            df_out['Season_Category'] = pd.cut(df_out['DOY'], bins=bins, labels=labels).astype(int)
            
        # Meteorological interaction: relative humidity * temperature (thermal humidity index proxy)
        if 'RH2M' in df_out.columns and 'T2M' in df_out.columns:
            df_out['RH_T_Interaction'] = (df_out['RH2M'] * df_out['T2M']) / 100.0
            
        return df_out

    def fit_preprocess(self, df_train):
        """Fits IQR capping and scaler on training split ONLY."""
        df_prep = self._engineer_features(df_train).copy()
        y_train = df_prep['Rain'].values
        
        drop_cols = ['Rain', 'Date', 'YEAR', 'DOY', 'DISTRICT', 'District', 'STATE_UT_NAME', 'State']
        feature_cols = [c for c in df_prep.columns if c not in drop_cols and pd.api.types.is_numeric_dtype(df_prep[c])]
        self.feature_names_in = feature_cols
        
        X_train_raw = df_prep[feature_cols].copy()
        
        # IQR Outlier Capping (Pathania et al.)
        X_capped = X_train_raw.copy()
        for col in feature_cols:
            vals = X_train_raw[col].dropna()
            q25, q75 = vals.quantile(0.25), vals.quantile(0.75)
            iqr = q75 - q25
            lower = q25 - 1.5 * iqr
            upper = q75 + 1.5 * iqr
            self.iqr_bounds[col] = (lower, upper)
            X_capped[col] = np.clip(X_train_raw[col], lower, upper)
            
        X_scaled = self.scaler.fit_transform(X_capped)
        return X_scaled, y_train

    def transform_preprocess(self, df_test):
        """Transforms test features using fitted IQR bounds and scaler."""
        df_prep = self._engineer_features(df_test).copy()
        y_test = df_prep['Rain'].values if 'Rain' in df_prep.columns else None
        
        X_test_raw = df_prep[self.feature_names_in].copy()
        X_capped = X_test_raw.copy()
        for col in self.feature_names_in:
            lower, upper = self.iqr_bounds.get(col, (-np.inf, np.inf))
            X_capped[col] = np.clip(X_test_raw[col], lower, upper)
            
        X_scaled = self.scaler.transform(X_capped)
        return X_scaled, y_test

    def fit(self, df_train):
        start_time = time.time()
        X_train, y_train = self.fit_preprocess(df_train)
        
        # Logistic Regression with balanced weighting
        self.model = LogisticRegression(
            penalty='l2', C=1.0, solver='lbfgs', max_iter=1000,
            class_weight='balanced', random_state=self.random_state
        )
        self.model.fit(X_train, y_train)
        self.train_time = time.time() - start_time
        return self

    def predict(self, df_test):
        X_test, _ = self.transform_preprocess(df_test)
        return self.model.predict(X_test)

    def predict_proba(self, df_test):
        X_test, _ = self.transform_preprocess(df_test)
        return self.model.predict_proba(X_test)[:, 1]

    def evaluate(self, df_test):
        X_test, y_test = self.transform_preprocess(df_test)
        y_pred = self.model.predict(X_test)
        y_prob = self.model.predict_proba(X_test)[:, 1]
        self.metrics = compute_metrics(y_test, y_pred, y_prob, self.train_time)
        return self.metrics
