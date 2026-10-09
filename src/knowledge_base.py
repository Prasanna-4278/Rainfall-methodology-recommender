"""
Methodology Knowledge Base Module
Synthesizes the empirical findings, algorithmic profiles, and dataset affinity rules
into a structured methodology knowledge base stored in methodology_profiles.csv.
"""
import os
import pandas as pd


def build_methodology_knowledge_base(benchmark_results_df=None, output_path=None):
    """
    Constructs methodology_profiles.csv linking dataset characteristics
    to methodology properties, benchmark metrics, and suitability conditions.
    """
    if output_path is None:
        output_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "methodology_profiles.csv")
        
    profiles = [
        {
            'methodology': 'Paper 1 (Talan 2026)',
            'paper': 'T. Talan (2026). Machine learning-based rainfall prediction across temporal scales: model benchmarking and explainability analysis',
            'missing_value_strategy': 'Sentinel detection (-999) + median imputation',
            'outlier_strategy': '3-standard-deviation bounding / winsorization',
            'feature_selection': 'Tree-based SelectFromModel importance thresholding',
            'scaling': 'StandardScaler fitted on training split',
            'temporal_features': 'Lag 1, 2, 3 + Rolling Means (3-day, 7-day) + DOY cyclics',
            'model_family': 'Tree Ensembles (RF, XGBoost, LightGBM, GradientBoosting) & MLP',
            'champion_model': 'Random Forest Classifier',
            'affinity_temporal_order': 'High (Requires continuous chronological sequence for lags/rolling)',
            'affinity_nonlinearity': 'Very High (Excels with complex non-linear meteorological interactions)',
            'affinity_sample_size': 'Moderate to Large (Needs sufficient observations for multi-day lag windows)',
            'affinity_class_imbalance': 'High (Balanced tree weighting mitigates rainfall event rarity)',
            'strengths': 'Captures multi-day weather momentum, atmospheric pressure dips, and humidity trends; SHAP explainability provides transparent meteorological attribution.',
            'weaknesses': 'Sensitive to temporal gaps/missing days; creates higher dimensionality requiring feature selection; slight latency overhead from lag computation.',
            'suitability_rule': 'Best when dataset has strong temporal continuity, high autocorrelation, non-linear weather interactions, and requires high Rain Recall.'
        },
        {
            'methodology': 'Paper 2 (Pathania et al. 2024)',
            'paper': 'Anuj Pathania et al. (2024). Harnessing Machine Learning for Precise Rainfall Forecasting: A Comparative Study',
            'missing_value_strategy': 'Sentinel detection + mean imputation',
            'outlier_strategy': 'Interquartile Range (IQR) 1.5x boundary capping',
            'feature_selection': 'Domain feature engineering (meteorological interaction & seasonal bins)',
            'scaling': 'StandardScaler fitted on training split',
            'temporal_features': 'Seasonal category bins (Winter, Pre-Monsoon, Monsoon, Post-Monsoon)',
            'model_family': 'Linear Models (Logistic Regression with calibrated probability)',
            'champion_model': 'Logistic Regression Classifier',
            'affinity_temporal_order': 'Low (Can operate on unordered or scattered daily records)',
            'affinity_nonlinearity': 'Moderate (Relies on explicitly engineered interaction terms)',
            'affinity_sample_size': 'Small to Moderate (Highly sample-efficient, low risk of overfitting)',
            'affinity_class_imbalance': 'Moderate (Class-weighted loss helps identify rainfall days)',
            'strengths': 'Highly interpretable linear coefficients, extremely fast execution, robust against overfitting on small or noisy datasets.',
            'weaknesses': 'Cannot autonomously capture complex high-order non-linearities without manual feature engineering; lower peak accuracy on intricate weather dynamics.',
            'suitability_rule': 'Best when dataset has small sample size, scattered non-consecutive dates, strict low-latency requirements, or linear meteorological regimes.'
        },
        {
            'methodology': 'Paper 3 (Kanmani Pappa et al. 2025)',
            'paper': 'C. Kanmani Pappa et al. (2025). Rainfall Prediction Using Machine Learning Techniques',
            'missing_value_strategy': 'Median imputation',
            'outlier_strategy': 'Normalization range bounding',
            'feature_selection': 'Domain correlation analysis (|r| >= 0.10 filtering against target Rain)',
            'scaling': 'MinMaxScaler [0, 1] normalization fitted on training split',
            'temporal_features': 'None (pure meteorological domain features)',
            'model_family': 'Linear & Tree Models (Logistic Regression, Random Forest)',
            'champion_model': 'Random Forest Classifier',
            'affinity_temporal_order': 'None (Invariant to temporal ordering)',
            'affinity_nonlinearity': 'High (Random Forest component handles non-linear boundaries)',
            'affinity_sample_size': 'Moderate (Correlation filtering eliminates noisy dimensions)',
            'affinity_class_imbalance': 'Moderate to High',
            'strengths': 'Prunes noisy or zero-correlation variables upfront; normalized features are safe for distance-based comparisons; straightforward and fast.',
            'weaknesses': 'Simple linear correlation threshold can prematurely discard features that only exhibit predictive power through non-linear interactions.',
            'suitability_rule': 'Best when dataset has redundant/weak predictor features, moderate sample size, no reliable time indexing, and strong individual feature correlations.'
        },
        {
            'methodology': 'Paper 4 (Khan et al. 2024)',
            'paper': 'Mohd Usman Saeed Khan et al. (2024). Comparative analysis of different rainfall prediction models: A case study of Aligarh City, India',
            'missing_value_strategy': 'Missing-value analysis + median imputation',
            'outlier_strategy': 'StandardScaler Gaussian standardization',
            'feature_selection': 'Dual-Stage: SelectKBest (ANOVA/Chi-square) + Random Forest SelectFromModel',
            'scaling': 'StandardScaler fitted on training split',
            'temporal_features': 'Date decomposition (DOY cyclic sin/cos + calendar quarter)',
            'model_family': 'Decision Tree, Logistic Regression, MLP, Random Forest',
            'champion_model': 'Random Forest Classifier',
            'affinity_temporal_order': 'Moderate (Uses calendar cyclics without requiring strict lag chains)',
            'affinity_nonlinearity': 'High (Dual-stage filtering preserves strong non-linear predictors)',
            'affinity_sample_size': 'Moderate to Large',
            'affinity_class_imbalance': 'High (Cohen Kappa and balanced weighting optimize classification balance)',
            'strengths': 'Rigorous dual-stage dimensionality reduction prevents curse of dimensionality; robust evaluation using Cohen Kappa; adapts well across tree and neural models.',
            'weaknesses': 'Dual-stage feature selection adds computational training time; may eliminate subtle marginal features in small datasets.',
            'suitability_rule': 'Best when dataset contains many candidate features (high dimensionality), moderate sample size, calendar periodicity, and requires strong Cohen Kappa performance.'
        }
    ]
    
    # If benchmark results dataframe available, attach empirical performance numbers
    if benchmark_results_df is not None:
        for p in profiles:
            p_name = p['methodology']
            match = benchmark_results_df[
                (benchmark_results_df['Methodology'] == p_name) &
                (benchmark_results_df['Model'].str.contains('Champion'))
            ]
            if len(match) > 0:
                row = match.iloc[0]
                p['benchmark_accuracy'] = row['Accuracy']
                p['benchmark_rain_precision'] = row['Rain Precision']
                p['benchmark_rain_recall'] = row['Rain Recall']
                p['benchmark_rain_f1'] = row['Rain F1']
                p['benchmark_roc_auc'] = row['ROC-AUC']
                p['benchmark_cohen_kappa'] = row['Cohen Kappa']
                p['benchmark_balanced_score'] = row['Balanced Score']
                
    df_kb = pd.DataFrame(profiles)
    df_kb.to_csv(output_path, index=False)
    
    # Also save in results/tables
    alt_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", "tables")
    os.makedirs(alt_dir, exist_ok=True)
    df_kb.to_csv(os.path.join(alt_dir, "methodology_profiles.csv"), index=False)
    
    return df_kb
