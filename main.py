"""
Main Execution Script for Comparative Methodology Recommendation System
Supports three operating modes:
  1. --mode benchmark : Executes common-dataset 4-methodology comparison, generates figures,
                        and compiles methodology_profiles.csv knowledge base.
  2. --mode recommend --data <path> : Profiles an unseen/new dataset, recommends the single most
                        suitable published methodology, and predicts rainfall.
  3. --mode test : Runs the complete end-to-end research workflow including unseen holdout demonstration.
"""
import os
import sys
import argparse
import subprocess
import pandas as pd

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import load_nasa_power_csv
from src.data_cleaning import clean_dataset
from src.dataset_profiler import profile_dataset
from src.comparison import run_benchmark_comparison
from src.knowledge_base import build_methodology_knowledge_base
from src.methodology_recommender import MethodologyRecommender
from src.prediction import execute_recommended_prediction


def run_benchmark_mode(data_path):
    print("=" * 80)
    print("STEP 1: RUNNING COMMON-DATASET METHODOLOGY BENCHMARK EXPERIMENT")
    print("=" * 80)
    print(f"Loading NASA POWER Hyderabad common dataset: {data_path}")
    
    # Run full comparative benchmark
    df_results, trained_pipelines, profile, synthesis_report = run_benchmark_comparison(data_path)
    
    print()
    print(synthesis_report)
    print("Benchmark results summary:")
    print(df_results[['Methodology', 'Model', 'Accuracy', 'Rain Recall', 'Rain F1', 'ROC-AUC', 'Balanced Score']].to_string(index=False))
    
    print()
    print("=" * 80)
    print("STEP 2: BUILDING METHODOLOGY KNOWLEDGE BASE")
    print("=" * 80)
    kb_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "methodology_profiles.csv")
    df_kb = build_methodology_knowledge_base(df_results, kb_path)
    print(f"Knowledge base successfully created at: {kb_path}")
    print(f"Recorded {len(df_kb)} methodology profiles with learned dataset affinity rules.")
    
    print()
    print("All 14 research figures generated in: results/figures/")
    print("Comparative tables saved in: results/tables/")
    return df_results, trained_pipelines


def run_recommend_mode(new_data_path, train_data_path=None):
    print("=" * 80)
    print("METHODOLOGY RECOMMENDATION & PREDICTION FOR UNSEEN DATASET")
    print("=" * 80)
    print(f"Target Unseen Dataset: {new_data_path}")
    
    # 1. Load target data
    if new_data_path.endswith('.csv'):
        # Check if NASA header exists
        with open(new_data_path, 'r', encoding='utf-8') as f:
            first_line = f.readline()
        if "-BEGIN HEADER-" in first_line:
            df_raw, _ = load_nasa_power_csv(new_data_path)
        else:
            df_raw = pd.read_csv(new_data_path)
            if 'YEAR' in df_raw.columns and 'DOY' in df_raw.columns and 'Date' not in df_raw.columns:
                df_raw['Date'] = pd.to_datetime(df_raw['YEAR'].astype(str) + '-' + df_raw['DOY'].astype(str), format='%Y-%j', errors='coerce')
    else:
        raise ValueError("Unsupported data format. Please provide a CSV file.")

    # 2. Clean data without target leakage
    if 'PRECTOTCORR' in df_raw.columns:
        df_clean, cleaning_log, _ = clean_dataset(df_raw, drop_unphysical_wd=True)
    else:
        df_clean = df_raw.copy()
        if 'WD10M' in df_clean.columns and (df_clean['WD10M'] < 0).all():
            df_clean = df_clean.drop(columns=['WD10M'])
            
    # 3. Profile Dataset Characteristics
    profile, profile_text = profile_dataset(df_clean, target_col='Rain' if 'Rain' in df_clean.columns else None)
    print(profile_text)
    
    # 4. Initialize Recommender and Evaluate Suitability
    kb_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "methodology_profiles.csv")
    recommender = MethodologyRecommender(kb_path)
    rec_result = recommender.recommend(df_clean)
    
    print()
    print(rec_result['report_text'])
    
    # 5. Apply Recommended Methodology for Rainfall Prediction
    print("=" * 80)
    print(f"APPLYING RECOMMENDED METHODOLOGY: {rec_result['recommended_methodology']}")
    print("=" * 80)
    
    # Load training partition from common benchmark dataset
    if train_data_path is None:
        train_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "raw", "POWER_Point_Daily_Hyderabad.csv")
        
    df_train_raw, _ = load_nasa_power_csv(train_data_path)
    df_train_clean, _, _ = clean_dataset(df_train_raw, drop_unphysical_wd=True)
    
    pred_output_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "results", "predictions", "rainfall_predictions.csv"
    )
    
    preds_df, eval_metrics = execute_recommended_prediction(
        df_train_clean, df_clean,
        rec_result['recommended_methodology'],
        output_path=pred_output_path
    )
    
    print()
    print("Sample Predictions Generated:")
    print(preds_df.head(10).to_string(index=False))
    
    if eval_metrics:
        print()
        print("Ground-Truth Validation on Unseen Dataset:")
        print(f"  • Accuracy      : {eval_metrics['accuracy']:.4f}")
        print(f"  • Rain Precision: {eval_metrics['precision']:.4f}")
        print(f"  • Rain Recall   : {eval_metrics['recall']:.4f}")
        print(f"  • Rain F1-score : {eval_metrics['f1']:.4f}")
        print(f"  • ROC-AUC       : {eval_metrics['roc_auc']}")
        print(f"  • Cohen's Kappa : {eval_metrics['cohen_kappa']:.4f}")
        
    print()
    print(f"Prediction pipeline complete. Results saved to: {pred_output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Comparative Methodology Recommendation System for Rainfall Prediction Using Research Paper Analysis"
    )
    parser.add_argument(
        '--mode', choices=['benchmark', 'recommend', 'test'], default='benchmark',
        help="Operating mode: 'benchmark' runs common comparison, 'recommend' profiles new data & predicts, 'test' runs full demo."
    )
    parser.add_argument(
        '--data', type=str, default=None,
        help="Path to new unseen dataset (required for --mode recommend)."
    )
    parser.add_argument(
        '--common_data', type=str, default=None,
        help="Path to common NASA POWER benchmark dataset (defaults to data/raw/POWER_Point_Daily_Hyderabad.csv)."
    )
    
    args = parser.parse_args()
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    common_data_path = args.common_data or os.path.join(base_dir, "data", "raw", "POWER_Point_Daily_Hyderabad.csv")
    
    if not os.path.exists(common_data_path):
        alt_path = os.path.join(base_dir, "data", "raw", "POWER_Point_Daily_20210101_20261001_017d39N_078d49E_LST(1).csv")
        if os.path.exists(alt_path):
            common_data_path = alt_path
            
    if args.mode == 'benchmark':
        run_benchmark_mode(common_data_path)
    elif args.mode == 'recommend':
        if not args.data:
            print("Error: --data <path_to_csv> is required for --mode recommend.")
            sys.exit(1)
        run_recommend_mode(args.data, common_data_path)
    elif args.mode == 'test':
        print("RUNNING COMPLETE END-TO-END VERIFICATION WORKFLOW")
        # Step A: Benchmark
        run_benchmark_mode(common_data_path)
        # Step B: Prepare holdout test dataset if not already present
        holdout_path = os.path.join(base_dir, "data", "test", "unseen_holdout_dataset.csv")
        if not os.path.exists(holdout_path):
            print("Generating unseen holdout test dataset...")
            subprocess.run([sys.executable, os.path.join(base_dir, "data", "test", "create_holdout.py")], check=True)
        # Step C: Recommend on holdout
        run_recommend_mode(holdout_path, common_data_path)


if __name__ == '__main__':
    main()
