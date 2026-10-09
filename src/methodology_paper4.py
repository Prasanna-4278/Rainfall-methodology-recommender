"""
Methodology 4: Adaptation of Mohd Usman Saeed Khan et al. (2024)
"Comparative analysis of different rainfall prediction models:
A case study of Aligarh City, India"

Key Pipeline Elements:
1. Date / temporal processing
2. Binary encoding
3. Missing-value analysis
4. Dual-stage feature selection:
   - SelectKBest (ANOVA / Chi-squared)
   - Random Forest feature importance + SelectFromModel
5. Correlation analysis
6. StandardScaler
7. Classifiers: Logistic Regression, Decision Tree, MLP, Random Forest
8. Comprehensive metrics: Acc, Prec, Rec, F1, ROC-AUC, Cohen's Kappa, Confusion Matrix
"""
import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, SelectFromModel
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

from src.evaluation import compute_metrics
from src.feature_engineering import add_cyclic_calendar_features


class KhanMethodologyAdaptation:
    def __init__(self, model_type='random_forest', random_state=42):
        self.name = "Paper 4 (Khan et al. 2024 Adaptation)"
        self.paper_citation = (
            "Mohd Usman Saeed Khan et al. (2024). Comparative analysis of different rainfall prediction models: "
            "A case study of Aligarh City, India."
        )
        self.model_type = model_type
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.kbest_selector = None
        self.rf_selector = None
        self.selected_feature_names = []
        self.model = None
        self.feature_names_in = []
        self.metrics = {}
        self.train_time = 0.0

    def _temporal_date_processing(self, df):
        """Processes calendar dates into cyclic and seasonal indicators."""
        df_out = df.copy()
        df_out = add_cyclic_calendar_features(df_out)
        if 'DOY' in df_out.columns:
            # Quarter indicator
            df_out['Quarter'] = ((df_out['DOY'] - 1) // 91 + 1).clip(1, 4)
        return df_out

    def fit_preprocess(self, df_train):
        """Fits date processing, dual-stage feature selection, and scaler on train split."""
        df_prep = self._temporal_date_processing(df_train).copy()
        y_train = df_prep['Rain'].values
        
        drop_cols = ['Rain', 'Date', 'YEAR', 'DOY', 'DISTRICT', 'District', 'STATE_UT_NAME', 'State']
        feature_cols = [c for c in df_prep.columns if c not in drop_cols and pd.api.types.is_numeric_dtype(df_prep[c])]
        self.feature_names_in = feature_cols
        
        X_train_raw = df_prep[feature_cols].fillna(df_prep[feature_cols].median())
        
        # 1. StandardScaler
        X_scaled = self.scaler.fit_transform(X_train_raw)
        
        # 2. Stage 1: SelectKBest (ANOVA F-statistic / Chi-2 analog for continuous)
        k = max(3, min(len(feature_cols), int(len(feature_cols) * 0.85)))
        self.kbest_selector = SelectKBest(score_func=f_classif, k=k)
        X_kbest = self.kbest_selector.fit_transform(X_scaled, y_train)
        
        # 3. Stage 2: SelectFromModel via Random Forest Feature Importance
        rf_importances = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=self.random_state)
        rf_importances.fit(X_kbest, y_train)
        self.rf_selector = SelectFromModel(rf_importances, threshold='mean', prefit=True)
        
        X_final = self.rf_selector.transform(X_kbest)
        
        # Track selected feature names
        kbest_names = [f for f, s in zip(feature_cols, self.kbest_selector.get_support()) if s]
        self.selected_feature_names = [f for f, s in zip(kbest_names, self.rf_selector.get_support()) if s]
        
        return X_final, y_train

    def transform_preprocess(self, df_test):
        """Transforms test set through the dual-stage selection pipeline."""
        df_prep = self._temporal_date_processing(df_test).copy()
        y_test = df_prep['Rain'].values if 'Rain' in df_prep.columns else None
        
        X_test_raw = df_prep[self.feature_names_in].fillna(df_prep[self.feature_names_in].median())
        X_scaled = self.scaler.transform(X_test_raw)
        X_kbest = self.kbest_selector.transform(X_scaled)
        X_final = self.rf_selector.transform(X_kbest)
        return X_final, y_test

    def fit(self, df_train):
        start_time = time.time()
        X_train, y_train = self.fit_preprocess(df_train)
        
        if self.model_type == 'logistic_regression':
            self.model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=self.random_state)
        elif self.model_type == 'decision_tree':
            self.model = DecisionTreeClassifier(max_depth=6, class_weight='balanced', random_state=self.random_state)
        elif self.model_type == 'mlp':
            self.model = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=400, random_state=self.random_state)
        else: # Default champion: Random Forest
            self.model = RandomForestClassifier(
                n_estimators=200, max_depth=8, class_weight='balanced',
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
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X_test)[:, 1]
        return None

    def evaluate(self, df_test):
        X_test, y_test = self.transform_preprocess(df_test)
        y_pred = self.model.predict(X_test)
        y_prob = self.model.predict_proba(X_test)[:, 1] if hasattr(self.model, 'predict_proba') else None
        self.metrics = compute_metrics(y_test, y_pred, y_prob, self.train_time)
        return self.metrics
