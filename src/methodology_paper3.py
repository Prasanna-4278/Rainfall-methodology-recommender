"""
Methodology 3: Adaptation of C. Kanmani Pappa et al. (2025)
"Rainfall Prediction Using Machine Learning Techniques"

Key Pipeline Elements:
1. Mean / median imputation
2. Normalization (MinMaxScaler)
3. Binary target encoding
4. Correlation analysis & domain-based feature selection
5. Logistic Regression & Random Forest
6. Comprehensive evaluation
"""
import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from src.evaluation import compute_metrics


class KanmaniPappaMethodologyAdaptation:
    def __init__(self, model_type='random_forest', corr_threshold=0.10, random_state=42):
        self.name = "Paper 3 (Kanmani Pappa et al. 2025 Adaptation)"
        self.paper_citation = (
            "C. Kanmani Pappa et al. (2025). Rainfall Prediction Using Machine Learning Techniques."
        )
        self.model_type = model_type
        self.corr_threshold = corr_threshold
        self.random_state = random_state
        self.imputer_medians = {}
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.selected_features = []
        self.model = None
        self.feature_names_in = []
        self.metrics = {}
        self.train_time = 0.0

    def fit_preprocess(self, df_train):
        """Imputation, MinMax normalization, and correlation filtering fitted on train."""
        df_prep = df_train.copy()
        y_train = df_prep['Rain'].values
        
        drop_cols = ['Rain', 'Date', 'YEAR', 'DOY', 'DISTRICT', 'District', 'STATE_UT_NAME', 'State']
        feature_cols = [c for c in df_prep.columns if c not in drop_cols and pd.api.types.is_numeric_dtype(df_prep[c])]
        self.feature_names_in = feature_cols
        
        # 1. Median Imputation
        X_train_raw = df_prep[feature_cols].copy()
        for col in feature_cols:
            med = X_train_raw[col].median()
            self.imputer_medians[col] = med
            X_train_raw[col] = X_train_raw[col].fillna(med)
            
        # 2. Correlation Analysis and Domain Feature Selection
        train_df_corr = X_train_raw.copy()
        train_df_corr['Rain'] = y_train
        correlations = train_df_corr.corr()['Rain'].drop('Rain').abs()
        
        # Select features with correlation >= threshold
        selected = correlations[correlations >= self.corr_threshold].index.tolist()
        if len(selected) == 0:
            # Fallback to top features if none exceed threshold
            selected = correlations.nlargest(3).index.tolist()
        self.selected_features = selected
        
        # 3. Normalization (MinMaxScaler)
        X_selected = X_train_raw[self.selected_features].values
        X_scaled = self.scaler.fit_transform(X_selected)
        return X_scaled, y_train

    def transform_preprocess(self, df_test):
        """Applies fitted medians, selected columns, and MinMax scaler to test set."""
        df_prep = df_test.copy()
        y_test = df_prep['Rain'].values if 'Rain' in df_prep.columns else None
        
        X_test_raw = df_prep[self.selected_features].copy()
        for col in self.selected_features:
            med = self.imputer_medians.get(col, 0.0)
            X_test_raw[col] = X_test_raw[col].fillna(med)
            
        X_scaled = self.scaler.transform(X_test_raw.values)
        return X_scaled, y_test

    def fit(self, df_train):
        start_time = time.time()
        X_train, y_train = self.fit_preprocess(df_train)
        
        if self.model_type == 'logistic_regression':
            self.model = LogisticRegression(
                C=1.0, class_weight='balanced', max_iter=1000, random_state=self.random_state
            )
        else: # Default champion: Random Forest
            self.model = RandomForestClassifier(
                n_estimators=150, max_depth=7, class_weight='balanced',
                random_state=self.random_state, n_jobs=-1
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
