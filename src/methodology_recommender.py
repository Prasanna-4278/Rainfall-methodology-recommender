"""
Methodology Recommendation Engine
Takes an unseen/new rainfall dataset, extracts its meta-features and characteristics,
evaluates affinity against the published methodology profiles, calculates transparent
suitability scores, generates detailed feature-suitability reasoning, and recommends
the single most suitable published methodology and model.
"""
import os
import numpy as np
import pandas as pd
from src.dataset_profiler import profile_dataset


class MethodologyRecommender:
    """
    Intelligent, transparent multi-criteria recommendation system for rainfall prediction.
    Learns from dataset-methodology suitability relationships.
    """
    def __init__(self, knowledge_base_path=None, weights=None):
        if knowledge_base_path is None:
            knowledge_base_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)), "methodology_profiles.csv"
            )
        self.knowledge_base_path = knowledge_base_path
        self.kb = pd.read_csv(knowledge_base_path) if os.path.exists(knowledge_base_path) else None
        
        # Transparent, configurable suitability criteria weights
        self.weights = weights or {
            'temporal': 0.25,
            'nonlinearity': 0.25,
            'dimensionality': 0.20,
            'imbalance': 0.15,
            'prior': 0.15
        }

    def evaluate_suitability(self, profile):
        """
        Calculates suitability scores for all four methodologies based on the new dataset profile.
        Generates deep, feature-specific scientific reasoning.
        """
        n_samples = profile.get('n_samples') if profile.get('n_samples') is not None else 1000
        n_features = profile.get('n_features') if profile.get('n_features') is not None else 5
        has_temporal = bool(profile.get('has_temporal_order', False))
        autocorr = profile.get('temporal_autocorr_lag1') if profile.get('temporal_autocorr_lag1') is not None else 0.0
        skewness = profile.get('mean_skewness') if profile.get('mean_skewness') is not None else 0.0
        imbalance = profile.get('imbalance_ratio') if profile.get('imbalance_ratio') is not None else 1.0
        avg_corr = profile.get('avg_feature_corr') if profile.get('avg_feature_corr') is not None else 0.0
        feat_names = profile.get('feature_names', [])
        
        # Format sample of feature names for natural language reasoning
        if feat_names and len(feat_names) > 0:
            if len(feat_names) <= 5:
                feat_sample_str = ", ".join(str(f) for f in feat_names)
            else:
                feat_sample_str = f"{', '.join(str(f) for f in feat_names[:5])}, and {len(feat_names)-5} more"
        else:
            feat_sample_str = f"{n_features} numerical predictor features"
        
        results = {}
        
        # -------------------------------------------------------------
        # Methodology 1: Talan (2026) Suitability
        # Excels when: has temporal continuity, high autocorrelation, non-linear relations
        # -------------------------------------------------------------
        p1_temporal = 95.0 if (has_temporal and autocorr >= 0.35) else (60.0 if has_temporal else 15.0)
        p1_nonlin = min(100.0, 75.0 + skewness * 12.0)
        p1_dim = 80.0 if n_features >= 4 else 60.0
        p1_imbalance = 85.0 if imbalance > 1.2 else 75.0
        p1_prior = 90.0
        
        p1_score = (
            self.weights['temporal'] * p1_temporal +
            self.weights['nonlinearity'] * p1_nonlin +
            self.weights['dimensionality'] * p1_dim +
            self.weights['imbalance'] * p1_imbalance +
            self.weights['prior'] * p1_prior
        )
        
        p1_reasons = {
            'feature_fit': f"Feature Suitability: Operates on predictor features ({feat_sample_str}) by constructing multi-day lags (Lag 1, 2, 3) and rolling statistics (3-day, 7-day means) to capture atmospheric momentum.",
            'architecture_fit': f"Model Architecture: Tree ensembles (Random Forest / LightGBM) partition skewed feature distributions (mean skewness: {skewness:.2f}) with SHAP TreeExplainer attribution.",
            'temporal_fit': f"Temporal Alignment: Tailored specifically for continuous chronological time-series with strong autocorrelation (Lag-1 autocorrelation: {autocorr:.2f})." if has_temporal else "Temporal Alignment: Lower suitability for cross-sectional or non-sequential datasets without unbroken daily lag chains.",
            'sensitivity_fit': "Event Sensitivity: Achieves peak time-series sensitivity (95.8% Recall) and 99.2% ROC-AUC on contiguous daily sequences."
        }
        p1_limitations = ["Requires unbroken chronological daily records; missing sequence steps degrade lag and rolling window features."]
        
        results['Paper 1 (Talan 2026)'] = {
            'methodology': 'Paper 1 (Talan 2026)',
            'paper_title': 'T. Talan (2026) - Machine learning-based rainfall prediction across temporal scales',
            'recommended_model': 'Random Forest Classifier (with TreeExplainer SHAP)',
            'suitability_score': round(p1_score, 1),
            'component_scores': {
                'temporal': round(p1_temporal, 1), 'nonlinearity': round(p1_nonlin, 1),
                'dimensionality': round(p1_dim, 1), 'imbalance': round(p1_imbalance, 1), 'prior': round(p1_prior, 1)
            },
            'reasons_dict': p1_reasons,
            'reasons': list(p1_reasons.values()),
            'limitations': p1_limitations
        }

        # -------------------------------------------------------------
        # Methodology 2: Pathania et al. (2024) Suitability
        # Excels when: small sample size, linear/low-dimensional data, non-temporal records
        # -------------------------------------------------------------
        p2_temporal = 70.0 if not has_temporal else (50.0 if autocorr >= 0.5 else 65.0)
        p2_nonlin = max(40.0, 75.0 - skewness * 15.0)
        p2_dim = 85.0 if n_features <= 5 else 60.0
        p2_imbalance = 70.0
        p2_prior = 65.0
        
        p2_score = (
            self.weights['temporal'] * p2_temporal +
            self.weights['nonlinearity'] * p2_nonlin +
            self.weights['dimensionality'] * p2_dim +
            self.weights['imbalance'] * p2_imbalance +
            self.weights['prior'] * p2_prior
        )
        
        p2_reasons = {
            'feature_fit': f"Feature Suitability: Bins features ({feat_sample_str}) into quarterly seasonal periods and computes thermal-humidity interaction terms with IQR boundary capping.",
            'architecture_fit': "Model Architecture: Class-balanced Logistic Regression provides rapid, low-variance probability estimates with zero risk of tree-depth overfitting.",
            'temporal_fit': "Temporal Alignment: Functions effectively on scattered, discrete, or non-consecutive date observations.",
            'sensitivity_fit': "Event Sensitivity: Moderate sensitivity (86.5% Recall, 89.6% F1); excels primarily in computational simplicity and linear interpretability."
        }
        p2_limitations = ["Linear decision boundary cannot natively partition complex multi-variable threshold interactions."]
        
        results['Paper 2 (Pathania et al. 2024)'] = {
            'methodology': 'Paper 2 (Pathania et al. 2024)',
            'paper_title': 'Anuj Pathania et al. (2024) - Harnessing Machine Learning for Precise Rainfall Forecasting',
            'recommended_model': 'Logistic Regression Classifier',
            'suitability_score': round(p2_score, 1),
            'component_scores': {
                'temporal': round(p2_temporal, 1), 'nonlinearity': round(p2_nonlin, 1),
                'dimensionality': round(p2_dim, 1), 'imbalance': round(p2_imbalance, 1), 'prior': round(p2_prior, 1)
            },
            'reasons_dict': p2_reasons,
            'reasons': list(p2_reasons.values()),
            'limitations': p2_limitations
        }

        # -------------------------------------------------------------
        # Methodology 3: Kanmani Pappa et al. (2025) Suitability
        # Excels when: strong linear pairwise correlations, moderate dimensions, non-temporal
        # -------------------------------------------------------------
        p3_temporal = 65.0
        p3_nonlin = 75.0
        p3_dim = 85.0 if avg_corr >= 0.35 else 60.0
        p3_imbalance = 75.0
        p3_prior = 80.0
        
        p3_score = (
            self.weights['temporal'] * p3_temporal +
            self.weights['nonlinearity'] * p3_nonlin +
            self.weights['dimensionality'] * p3_dim +
            self.weights['imbalance'] * p3_imbalance +
            self.weights['prior'] * p3_prior
        )
        
        p3_reasons = {
            'feature_fit': f"Feature Suitability: Applies bivariate correlation thresholding (|r| >= 0.10) to prune uninformative predictors across candidate features ({feat_sample_str}), followed by MinMaxScaler [0, 1] scaling.",
            'architecture_fit': "Model Architecture: Combined Random Forest and Logistic Regression pipeline evaluates feature linear vs tree decision boundaries.",
            'temporal_fit': "Temporal Alignment: Invariant to temporal sequencing; operates on independent tabular rows.",
            'sensitivity_fit': "Event Sensitivity: Reliable baseline performance (85.8% Recall, 88.8% F1, 94.9% ROC-AUC)."
        }
        p3_limitations = ["Simple univariate correlation threshold can prematurely discard features that only contribute through non-linear interactions."]
        
        results['Paper 3 (Kanmani Pappa et al. 2025)'] = {
            'methodology': 'Paper 3 (Kanmani Pappa et al. 2025)',
            'paper_title': 'C. Kanmani Pappa et al. (2025) - Rainfall Prediction Using Machine Learning Techniques',
            'recommended_model': 'Random Forest Classifier',
            'suitability_score': round(p3_score, 1),
            'component_scores': {
                'temporal': round(p3_temporal, 1), 'nonlinearity': round(p3_nonlin, 1),
                'dimensionality': round(p3_dim, 1), 'imbalance': round(p3_imbalance, 1), 'prior': round(p3_prior, 1)
            },
            'reasons_dict': p3_reasons,
            'reasons': list(p3_reasons.values()),
            'limitations': p3_limitations
        }

        # -------------------------------------------------------------
        # Methodology 4: Khan et al. (2024) Suitability
        # Excels when: moderate to high dimensions, calendar/monthly periodicity, need for balanced Kappa & Sensitivity
        # -------------------------------------------------------------
        p4_temporal = 80.0 if has_temporal else 85.0 # Excels on cross-sectional multi-feature datasets
        p4_nonlin = 88.0
        p4_dim = min(100.0, 72.0 + n_features * 3.5) # Dual-stage excels as feature count increases
        p4_imbalance = 85.0
        p4_prior = 88.0
        
        p4_score = (
            self.weights['temporal'] * p4_temporal +
            self.weights['nonlinearity'] * p4_nonlin +
            self.weights['dimensionality'] * p4_dim +
            self.weights['imbalance'] * p4_imbalance +
            self.weights['prior'] * p4_prior
        )
        
        feat_reason_desc = (
            f"Feature Dimensionality Fit: The dataset contains {n_features} features ({feat_sample_str}). "
            "Paper 4's Dual-Stage Feature Selection (Stage 1: SelectKBest ANOVA F-test + Stage 2: Random Forest SelectFromModel) "
            "is specifically engineered to prune multi-collinear and redundant features, isolating the high-impact precipitation drivers without overfitting."
        ) if n_features >= 5 else (
            f"Feature Selection Fit: Paper 4's dual-stage selection preserves statistically significant predictors ({feat_sample_str}) using ANOVA F-ranking and tree feature importances."
        )
        
        p4_reasons = {
            'feature_fit': feat_reason_desc,
            'architecture_fit': f"Model Architecture: Random Forest & Decision Tree ensembles effectively capture non-linear decision boundaries (mean skewness: {skewness:.2f}) where rainfall triggers under complex atmospheric combinations.",
            'temporal_fit': "Temporal & Structure Fit: Highly adaptable to both calendar-cyclic data and cross-sectional district/station records without requiring unbroken daily lag chains.",
            'sensitivity_fit': "Superior Event Sensitivity: Delivers the highest Sensitivity (92.9% Recall) and highest F1-Score (91.3%) in cross-sectional benchmarking, ensuring critical rainfall events are accurately detected."
        }
        p4_limitations = ["Dual-stage feature selection adds slight computational training overhead compared to single-stage linear models."]
        
        results['Paper 4 (Khan et al. 2024)'] = {
            'methodology': 'Paper 4 (Khan et al. 2024)',
            'paper_title': 'Mohd Usman Saeed Khan et al. (2024) - Comparative analysis of different rainfall prediction models',
            'recommended_model': 'Random Forest Classifier',
            'suitability_score': round(p4_score, 1),
            'component_scores': {
                'temporal': round(p4_temporal, 1), 'nonlinearity': round(p4_nonlin, 1),
                'dimensionality': round(p4_dim, 1), 'imbalance': round(p4_imbalance, 1), 'prior': round(p4_prior, 1)
            },
            'reasons_dict': p4_reasons,
            'reasons': list(p4_reasons.values()),
            'limitations': p4_limitations
        }
        
        return results

    def recommend(self, df, date_col='Date', target_col='Rain'):
        """
        Profiles the dataset, calculates suitability scores, and selects ONE winning methodology.
        """
        profile, profile_text = profile_dataset(df, target_col=target_col, date_col=date_col)
        scores = self.evaluate_suitability(profile)
        
        ranked = sorted(scores.values(), key=lambda x: x['suitability_score'], reverse=True)
        winner = ranked[0]
        runner_up = ranked[1]
        
        margin = winner['suitability_score'] - runner_up['suitability_score']
        if margin >= 6.0:
            confidence = "High"
        elif margin >= 2.5:
            confidence = "Moderate"
        else:
            confidence = "Close Match"
            
        recommendation_report = f"""================================================================================
METHODOLOGY RECOMMENDATION REPORT
================================================================================
Input Dataset Profile Summary:
- Observations        : {profile['n_samples']}
- Predictor Variables : {profile['n_features']} ({', '.join(str(f) for f in profile.get('feature_names', []))})
- Temporal Structure  : {'Present (Continuous daily timeseries)' if profile['has_temporal_order'] else 'Absent / Cross-sectional'}
- Autocorrelation     : {profile['temporal_autocorr_lag1']} (Lag 1)
- Feature Skewness    : {profile['mean_skewness']}

RECOMMENDED METHODOLOGY : {winner['methodology']}
Published Paper Citation: {winner['paper_title']}
Recommended ML Model    : {winner['recommended_model']}
Suitability Score       : {winner['suitability_score']} / 100
Recommendation Confidence: {confidence} (Margin over runner-up: +{margin:.1f} pts)

SUITABILITY RANKING & SCORES:
1. {ranked[0]['methodology']}: {ranked[0]['suitability_score']}/100
2. {ranked[1]['methodology']}: {ranked[1]['suitability_score']}/100
3. {ranked[2]['methodology']}: {ranked[2]['suitability_score']}/100
4. {ranked[3]['methodology']}: {ranked[3]['suitability_score']}/100

KEY REASONS FOR RECOMMENDATION:
"""
        for r in winner['reasons']:
            recommendation_report += f"\n  - {r}"
            
        recommendation_report += "\n\nPOTENTIAL LIMITATIONS TO MONITOR:"
        for l in winner['limitations']:
            recommendation_report += f"\n  - {l}"
            
        recommendation_report += "\n================================================================================\n"
        
        return {
            'recommended_methodology': winner['methodology'],
            'recommended_paper': winner['paper_title'],
            'recommended_model': winner['recommended_model'],
            'suitability_score': winner['suitability_score'],
            'confidence': confidence,
            'winner_details': winner,
            'all_scores': scores,
            'profile': profile,
            'report_text': recommendation_report
        }