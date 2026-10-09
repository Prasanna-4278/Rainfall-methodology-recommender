# A Comparative Methodology Recommendation System for Rainfall Prediction Using Research Paper Analysis

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Research: Applied Meteorology](https://img.shields.io/badge/Domain-Hydrometeorology-brightgreen.svg)]()

---

## 1. Project Objective & Core Research Paradigm

Traditional machine learning studies in rainfall forecasting routinely evaluate a cluster of algorithms on an arbitrary dataset, declare the model with the highest test accuracy as "superior," and stop there. 

This project operates on an entirely different research thesis:
> **Different published methodologies possess distinct inductive biases, preprocessing strategies, and dimensional requirements. The performance of a methodology is intrinsically governed by the underlying characteristics of the dataset.**

The objective is **not** to blindly run multiple models on a new dataset and pick whichever happens to yield the highest metric. Doing so creates massive computational redundancy and ignores the scientific rationale behind meteorological modeling.

Instead, this system implements an intelligent **dataset-to-methodology recommendation framework**:

```
        Four Published Methodologies
                      ↓
    Common Benchmark Dataset Experiments (NASA POWER)
                      ↓
  Methodology Performance + Dataset Characteristics Analysis
                      ↓
        Methodology Knowledge Base (methodology_profiles.csv)
                      ↓
           =======================
           NEW / UNSEEN DATASET
           =======================
                      ↓
         Dataset Profiling & Meta-Feature Extraction
                      ↓
         Multi-Criteria Suitability Matching
                      ↓
      RECOMMEND ONE PUBLISHED METHODOLOGY & MODEL
                      ↓
      Execute Rainfall Predictions with Probabilities
```

---

## 2. Four Published Methodologies (Adaptations)

Each methodology is implemented as an independent, modular pipeline adapting the author's published principles to the common binary daily precipitation classification task:

### Paper 1: T. Talan (2026)
* **Title**: *Machine learning-based rainfall prediction across temporal scales: model benchmarking and explainability analysis*
* **Core Principles**: Temporal dynamic modeling, lag structures, rolling statistics, outlier clipping, standardization, tree ensembles, and explainability.
* **Our Adaptation**:
  * Missing value imputation via median.
  * Outlier handling: $3\sigma$ (3-standard-deviation) boundary clipping on training features.
  * Strict chronological temporal feature engineering: Lag 1, Lag 2, Lag 3 days; Rolling Means over 3-day and 7-day windows (shifted by 1 day to prevent future leakage).
  * Feature standardization (`StandardScaler`).
  * Feature selection: `SelectFromModel` using Random Forest feature importance.
  * Model families: Random Forest (Champion), Gradient Boosting, XGBoost, LightGBM, MLP.
  * Explainability: SHAP (`shap.TreeExplainer`) summary plot.

### Paper 2: Anuj Pathania et al. (2024)
* **Title**: *Harnessing Machine Learning for Precise Rainfall Forecasting: A Comparative Study*
* **Core Principles**: Robust statistical outlier trimming, domain-driven seasonal feature engineering, and calibrated probabilistic classification.
* **Our Adaptation**:
  * Missing value handling.
  * Outlier capping via Interquartile Range (IQR): $[Q_1 - 1.5 \times \text{IQR}, Q_3 + 1.5 \times \text{IQR}]$.
  * Domain feature engineering: Astronomical seasonal binning (Winter, Pre-Monsoon, Monsoon, Post-Monsoon) + Relative Humidity $\times$ Temperature interaction index.
  * Model: `LogisticRegression` with class-balanced weighting and calibrated probability predictions.

### Paper 3: C. Kanmani Pappa et al. (2025)
* **Title**: *Rainfall Prediction Using Machine Learning Techniques*
* **Core Principles**: Normalized feature scaling, correlation analysis, and domain filtering.
* **Our Adaptation**:
  * Missing value imputation: Feature-wise median.
  * Normalization: `MinMaxScaler` mapping variables into $[0, 1]$.
  * Feature selection: Bivariate correlation thresholding ($|r| \ge 0.10$ with target `Rain`) to discard uninformative predictors.
  * Models: Logistic Regression and Random Forest Classifier (Champion).

### Paper 4: Mohd Usman Saeed Khan et al. (2024)
* **Title**: *Comparative analysis of different rainfall prediction models: A case study of Aligarh City, India*
* **Core Principles**: Date decomposition, dual-stage feature selection, Gaussian standardization, and balanced classification evaluation via Cohen's Kappa.
* **Our Adaptation**:
  * Date/temporal processing: Day-of-Year trigonometric cyclics ($\\sin, \\cos$) and calendar quarter.
  * Missing-value analysis and median imputation.
  * **Dual-Stage Feature Selection**:
    1. Stage 1: `SelectKBest` with ANOVA F-statistic ($f\_classif$).
    2. Stage 2: `SelectFromModel` using Random Forest feature importance thresholding.
  * Feature standardization (`StandardScaler`).
  * Models: Decision Tree, Logistic Regression, MLP Neural Net, and Random Forest Classifier (Champion).
  * Comprehensive evaluation including Cohen's Kappa.

---

## 3. Common Dataset & Data Quality Audit

* **Source**: NASA Prediction of Worldwide Energy Resources (POWER) Daily Meteorological Dataset.
* **Location**: Hyderabad, Telangana, India ($17.385^\circ\text{N}, 78.4867^\circ\text{E}$).
* **Period**: 2021-01-01 through 2026-10-01 (2,100 daily records).
* **Parameters**:
  * `T2M`: Temperature at 2 Meters ($^\circ\text{C}$)
  * `RH2M`: Relative Humidity at 2 Meters ($\%$)
  * `PS`: Surface Pressure ($	ext{kPa}$)
  * `WS10M`: Wind Speed at 10 Meters ($	ext{m/s}$)
  * `WD10M`: Wind Direction at 10 Meters ($	ext{Degrees}$)
  * `PRECTOTCORR`: Precipitation Corrected ($	ext{mm/day}$)

### Data Inspection & Critical Cleaning Decisions
1. **Trailing Sentinel Rows**: The final 3 rows (DOY 272–274 of 2026) contain `-999.0` for all meteorological variables and precipitation. In NASA POWER, `-999` denotes missing/uncomputed near-real-time data. These 3 unfinalized rows were dropped, leaving 2,097 valid daily records.
2. **WD10M Domain Diagnosis**:
   * Physical wind direction must lie within $[0^\circ, 360^\circ]$.
   * In the raw export, **100% of non-missing WD10M measurements are negative**, ranging from $-5,405^\circ$ to $-98,710^\circ$ (mean: $-8,132^\circ$, standard deviation: $7,608^\circ$).
   * These negative numbers represent internal fill codes or corrupted vector-averaging sentinel flags from MERRA-2.
   * **Domain Decision**: Feeding raw values with extreme negative magnitude and massive variance into machine learning models would severely distort scalers, gradient descent, and distance metrics. Hence, `WD10M` is cleanly dropped from predictive features and logged in the dataset profile audit trail.
3. **Target Derivation & Strict Non-Leakage**:
   $$\text{Rain} = \begin{cases} 1 & \text{if } \text{PRECTOTCORR} > 0 \\ 0 & \text{if } \text{PRECTOTCORR} = 0 \end{cases}$$
   * **Leakage Prevention**: `PRECTOTCORR` is **immediately dropped from the predictor matrix**. All models predict `Rain` using solely the atmospheric state variables (`T2M`, `RH2M`, `PS`, `WS10M`) and derived temporal features.

---

## 4. Fair Experimental Protocol

* **Strict Chronological Train/Test Split**:
  * **Train Period**: 2021-01-01 to 2024-12-31 ($N = 1,461$ days, $\approx 70\%$).
  * **Test Period**: 2025-01-01 to 2026-09-28 ($N = 636$ days, $\approx 30\%$).
  * *Zero future data leakage*: Random shuffling is prohibited.
* **Preprocessing Isolation**:
  * All scalers (`StandardScaler`, `MinMaxScaler`), outlier clipping bounds ($3\sigma$, IQR), imputation medians, and feature selectors (`SelectFromModel`, `SelectKBest`) are **fitted strictly on the training partition**. The test partition is purely transformed.

---

## 5. Experimental Benchmark Results

### Comprehensive Methodology Comparison Table

| Methodology | Model | Preprocessing | Feature Selection | Accuracy | Rain Precision | Rain Recall | Rain F1 | ROC-AUC | Cohen Kappa | Balanced Score | Train Time (s) |
|---|---|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Paper 1 (Talan 2026)** | Random Forest (Champion) | $3\sigma$ + Lags(1-3) + Rolling(3,7d) + StandardScaler | Tree SelectFromModel | 0.8752 | 0.9122 | 0.8760 | 0.8937 | 0.9364 | 0.7423 | 0.8942 | 0.620 |
| **Paper 1 (Talan 2026)** | Gradient Boosting | $3\sigma$ + Lags(1-3) + Rolling(3,7d) + StandardScaler | Tree SelectFromModel | 0.8784 | 0.9126 | 0.8813 | 0.8966 | 0.9477 | 0.7490 | 0.8996 | 0.690 |
| **Paper 1 (Talan 2026)** | XGBoost | $3\sigma$ + Lags(1-3) + Rolling(3,7d) + StandardScaler | Tree SelectFromModel | 0.8736 | 0.9098 | 0.8760 | 0.8925 | 0.9464 | 0.7391 | 0.8956 | 0.160 |
| **Paper 1 (Talan 2026)** | LightGBM | $3\sigma$ + Lags(1-3) + Rolling(3,7d) + StandardScaler | Tree SelectFromModel | 0.8673 | 0.9042 | 0.8707 | 0.8871 | 0.9466 | 0.7262 | 0.8913 | 0.080 |
| **Paper 1 (Talan 2026)** | MLP Neural Net | $3\sigma$ + Lags(1-3) + Rolling(3,7d) + StandardScaler | Tree SelectFromModel | 0.8768 | 0.9015 | 0.8918 | 0.8966 | 0.9272 | 0.7441 | 0.8990 | 0.720 |
| **Paper 2 (Pathania 2024)** | Logistic Regression (Champion) | IQR Capping + Seasonal Bins + RH-T interaction | Domain Feature Set | 0.8805 | 0.9298 | 0.8654 | 0.8962 | **0.9570** | 0.7554 | 0.8960 | 0.015 |
| **Paper 3 (Kanmani Pappa 2025)**| Random Forest (Champion) | Median Impute + MinMaxScaler | Corr Filter (\|r\| $\ge$ 0.10) | 0.8522 | 0.8992 | 0.8470 | 0.8723 | 0.9376 | 0.6974 | 0.8745 | 0.420 |
| **Paper 3 (Kanmani Pappa 2025)**| Logistic Regression | Median Impute + MinMaxScaler | Corr Filter (\|r\| $\ge$ 0.10) | 0.8711 | 0.9209 | 0.8575 | 0.8880 | 0.9493 | 0.7358 | 0.8879 | 0.010 |
| **Paper 4 (Khan et al. 2024)** | Random Forest (Champion) | Date Cyclic + StandardScaler | Dual-Stage (ANOVA + RF) | 0.8915 | 0.9237 | 0.8918 | 0.9074 | 0.9410 | 0.7766 | 0.9071 | 0.710 |
| **Paper 4 (Khan et al. 2024)** | Decision Tree | Date Cyclic + StandardScaler | Dual-Stage (ANOVA + RF) | **0.8947** | 0.8979 | **0.9288** | **0.9131** | 0.9122 | **0.7797** | 0.9166 | 0.012 |
| **Paper 4 (Khan et al. 2024)** | MLP Neural Net | Date Cyclic + StandardScaler | Dual-Stage (ANOVA + RF) | 0.8931 | 0.9040 | 0.9182 | 0.9110 | 0.9454 | 0.7770 | **0.9186** | 0.810 |
| **Paper 4 (Khan et al. 2024)** | Logistic Regression | Date Cyclic + StandardScaler | Dual-Stage (ANOVA + RF) | 0.8695 | **0.9353** | 0.8391 | 0.8846 | 0.9411 | 0.7335 | 0.8784 | 0.012 |

### Performance Synthesis
* **Best Accuracy**: Paper 4 (Khan et al. 2024) - Decision Tree (**89.47%**)
* **Best Rain Precision**: Paper 4 (Khan et al. 2024) - Logistic Regression (**93.53%**)
* **Best Rain Recall**: Paper 4 (Khan et al. 2024) - Decision Tree (**92.88%**)
* **Best Rain F1-Score**: Paper 4 (Khan et al. 2024) - Decision Tree (**91.31%**)
* **Best ROC-AUC**: Paper 2 (Pathania et al. 2024) - Logistic Regression (**0.9570**)
* **Best Overall Balanced Methodology**: Paper 4 (Khan et al. 2024) - MLP Neural Net (**Score: 0.9186**) followed by Paper 4 Random Forest (**0.9071**) and Paper 1 Gradient Boosting (**0.8996**).

---

## 6. Dataset Characteristics & Meta-Features

The system computes a rich meta-feature profile capturing the statistical and physical properties of the dataset:

```text
==================================================
DATASET PROFILE & CHARACTERISTICS
==================================================
Samples (N)             : 2097
Features (D)            : 4 (T2M, RH2M, PS, WS10M)
Sample-to-Feature Ratio : 524.2
Missingness             : 0.0%
Outliers (3-sigma)      : 0.38%
Outliers (IQR)          : 1.12%
Rain Events             : 56.8%
No-Rain Events          : 43.2%
Class Imbalance Ratio   : 1.31:1
Mean Feature Skewness   : 0.542
Average Pairwise Corr   : 0.457
Max Pairwise Corr       : 0.577
Temporal Continuity     : Yes
Autocorrelation (Lag 1) : 0.884
Seasonality Strength    : 1.000
Rainfall Variability CV : 2.350
==================================================
```

---

## 7. Methodology Knowledge Base (`methodology_profiles.csv`)

The knowledge base bridges dataset meta-features with methodology suitability:

| Methodology | Model Family | Missing Strategy | Outlier Strategy | Temporal Features | Strengths | Suitable Dataset Regime |
|---|---|---|---|---|---|---|
| **Paper 1 (Talan 2026)** | Tree Ensembles & MLP | Median | $3\sigma$ bounding | Lags 1-3, Rolling 3d, 7d | Atmospheric inertia, momentum dips, SHAP explainability | High temporal continuity, high autocorrelation ($>0.35$), multi-day weather series |
| **Paper 2 (Pathania 2024)** | Linear (Logistic Regression) | Mean | IQR 1.5x capping | Seasonal categories | Fast inference, robust to overfitting, low variance | Small datasets, scattered non-continuous records, linear dynamics |
| **Paper 3 (Kanmani Pappa 2025)** | LR & Random Forest | Median | MinMax scaling | None (Domain only) | Prunes noisy features upfront via $\|r\| \ge 0.10$ | Moderate sample size, high individual feature correlation, non-temporal records |
| **Paper 4 (Khan et al. 2024)** | DT, LR, MLP, Random Forest | Median | StandardScaler | Cyclic calendar encodings | Dual-stage selection eliminates collinearity, strong Cohen Kappa | Moderate-to-high dimensions, seasonal periodicity, non-linear feature sets |

---

## 8. Transparent Methodology Recommender

When presented with a **new, unseen dataset**, the recommender does **not** cheat by executing all four pipelines. Instead, it extracts the meta-features of the new dataset and evaluates suitability using a multi-criteria scoring function:

$$S_m = w_{\text{temp}} \cdot T_m + w_{\text{nonlin}} \cdot N_m + w_{\text{dim}} \cdot D_m + w_{\text{imbal}} \cdot B_m + w_{\text{prior}} \cdot P_m$$

Where the default, fully documented weights are:
* **$w_{\text{temp}} = 0.25$ (Temporal Affinity)**: Evaluates whether time ordering exists and if lag-1 autocorrelation is substantial.
* **$w_{\text{nonlin}} = 0.25$ (Non-Linearity Affinity)**: Assesses feature skewness and interaction complexity favoring tree ensembles over linear models.
* **$w_{\text{dim}} = 0.20$ (Dimensionality & Selection Affinity)**: Assesses predictor count and collinearity, favoring dual-stage or correlation filtering.
* **$w_{\text{imbal}} = 0.15$ (Class Imbalance Affinity)**: Evaluates the need for cost-sensitive loss functions to maintain high Rain Recall.
* **$w_{\text{prior}} = 0.15$ (Empirical Benchmark Prior)**: Incorporates the baseline performance established during the common research experiment.

### Demonstration on Unseen Holdout Scenario (`data/test/unseen_holdout_dataset.csv`):
* **Dataset characteristics**: $N = 271$ observations, daily continuous sequence, lag-1 autocorrelation $= 0.951$, seasonality strength $= 1.0$.
* **Recommender Output**:
  * **Recommended Methodology**: **Paper 1 (Talan 2026)**
  * **Recommended Model**: **Random Forest Classifier (with TreeExplainer SHAP)**
  * **Suitability Score**: **86.5 / 100** (Confidence: Moderate, Margin over runner-up: +3.0 pts)
  * **Rationale**: Exceptionally high autocorrelation ($0.95$) makes multi-day lag and rolling feature engineering highly effective for capturing rainfall momentum.
* **Prediction Results on Unseen Data**:
  * Accuracy: **93.28%**
  * Rain Precision: **93.57%**
  * Rain Recall: **95.81%**
  * Rain F1-Score: **94.67%**
  * ROC-AUC: **0.9921**
  * Cohen's Kappa: **0.8559**

---

## 9. Research Visualizations

All 14 publication-grade figures are stored in `results/figures/`:

1. `01_rainfall_distribution.png`: Histogram and KDE of positive precipitation events.
2. `02_class_distribution.png`: Binary class distribution (Rain vs No-Rain).
3. `03_correlation_heatmap.png`: Pearson correlation matrix among predictors and target.
4. `04_missing_values.png`: Percentage of NASA POWER $-999$ sentinel flags across parameters.
5. `05_outlier_boxplots.png`: Outlier distributions across temperature, humidity, pressure, and wind.
6. `06_feature_distributions.png`: Kernel density estimates stratified by Rain class.
7. `07_methodology_accuracy.png`: Test accuracy comparison across all 12 model pipelines.
8. `08_rain_recall_comparison.png`: Rain event recall comparison for the 4 champion pipelines.
9. `09_rain_f1_comparison.png`: Rain F1-score comparison for the 4 champions.
10. `10_roc_curves.png`: Multi-model ROC curves with respective AUC scores.
11. `11_confusion_matrices.png`: Confusion matrices for the four methodology champions.
12. `12_feature_importance.png`: Feature importance breakdown from the Khan et al. Random Forest pipeline.
13. `13_shap_summary_paper1.png`: SHAP TreeExplainer feature attribution beeswarm summary for Paper 1.
14. `14_suitability_comparison.png`: Multi-criteria performance radar comparison.

---

## 10. Project Directory Structure

```text
rainfall_methodology_recommender/
│
├── data/
│   ├── raw/
│   │   ├── POWER_Point_Daily_Hyderabad.csv
│   │   └── POWER_Point_Daily_20210101_20261001_017d39N_078d49E_LST(1).csv
│   └── test/
│       ├── create_holdout.py
│       └── unseen_holdout_dataset.csv
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py              # NASA POWER parser & date constructor
│   ├── data_cleaning.py            # -999 replacement, WD10M diagnosis, non-leakage target
│   ├── dataset_profiler.py         # Statistical & time-series meta-feature extraction
│   ├── feature_engineering.py      # Lags (1-3), rolling means (3,7d), cyclic DOY encodings
│   ├── methodology_paper1.py       # Talan (2026) adaptation (Temporal + RF/XGB/LGBM/MLP + SHAP)
│   ├── methodology_paper2.py       # Pathania et al. (2024) adaptation (IQR + Logistic Regression)
│   ├── methodology_paper3.py       # Kanmani Pappa et al. (2025) adaptation (MinMax + Corr filter + RF)
│   ├── methodology_paper4.py       # Khan et al. (2024) adaptation (Dual-Stage Selection + DT/MLP/RF)
│   ├── evaluation.py               # Comprehensive metrics (Acc, Prec, Rec, F1, ROC-AUC, Kappa)
│   ├── comparison.py               # Benchmark runner & visualizer (14 figures)
│   ├── knowledge_base.py           # Builds methodology_profiles.csv
│   ├── methodology_recommender.py  # Meta-feature suitability engine & recommender
│   └── prediction.py               # Inference pipeline for recommended methodology
│
├── results/
│   ├── figures/                    # 14 publication-grade figures (.png)
│   ├── tables/                     # Benchmark CSVs, profiles, and synthesis reports
│   └── predictions/                # Output predictions with probabilities (.csv)
│
├── notebooks/
│   └── comparative_study.ipynb     # Interactive Jupyter notebook walkthrough
│
├── main.py                         # CLI entry point (--mode benchmark | recommend | test)
├── requirements.txt                # Dependencies (scikit-learn, xgboost, lightgbm, shap, etc.)
├── methodology_profiles.csv        # Knowledge base exported to project root
└── README.md                       # Comprehensive project documentation
```

---

## 11. How to Run

### Installation
```bash
git clone <repository_url>
cd rainfall_methodology_recommender
pip install -r requirements.txt
```

### 1. Run the Common Benchmark Comparison
Executes all four methodologies on the common NASA POWER dataset, generates all 14 figures, and exports the knowledge base:
```bash
python main.py --mode benchmark
```

### 2. Recommend & Predict for a New Unseen Dataset
Profiles an unseen dataset, evaluates suitability, recommends the single best published methodology, and predicts rainfall:
```bash
python main.py --mode recommend --data data/test/unseen_holdout_dataset.csv
```

### 3. End-to-End Verification Test
Runs the complete research benchmark, updates the knowledge base, and executes the recommendation workflow on the unseen holdout dataset:
```bash
python main.py --mode test
```

---

## 12. Research Integrity Statement

1. **No Target Leakage**: The target variable `Rain` is derived from `PRECTOTCORR`. To uphold scientific rigor, `PRECTOTCORR` is strictly removed from all predictor sets.
2. **Explicit Labeling of Adaptations**: The published methodologies are adapted from different local geographies or regression formulations to a common daily classification benchmark. They are explicitly described as disciplined adaptations rather than exact algorithmic clones.
3. **No Retrospective Cheating**: The recommendation system does not execute all candidate pipelines on unseen data to choose the winner ex post facto; it infers suitability strictly from the learned relationship between dataset characteristics and methodology strengths.
