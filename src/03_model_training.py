"""
DSC5001 Project: Hong Kong AQHI High-Risk Hour Early Warning
=============================================================
Step 3: Model training, evaluation, and comparison

Models:
  1. Logistic Regression (baseline, L2 regularization)
  2. Random Forest
  3. XGBoost (gradient boosting)

Evaluation:
  - TimeSeriesSplit cross-validation
  - Recall, Precision, F1, PR-AUC
  - Confusion matrices
  - Threshold optimization
  - Cost-benefit analysis (missed alerts vs false alarms)
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    precision_score, recall_score, f1_score, 
    precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report
)
import xgboost as xgb

warnings.filterwarnings('ignore')

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'output'))
FIGURES_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'figures'))
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# Feature columns
FEATURE_COLS = [
    'hour', 'dayofweek', 'month', 'dayofyear',
    'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'month_sin', 'month_cos',
    'is_weekend',
    'AQHI_lag_1', 'AQHI_lag_2', 'AQHI_lag_3', 'AQHI_lag_6', 'AQHI_lag_12', 'AQHI_lag_24',
    'AQHI_rolling_mean_3', 'AQHI_rolling_mean_6', 'AQHI_rolling_std_6',
    'AQHI_diff_1', 'AQHI_diff_3',
    'Temp', 'RH', 'Rainfall', 'WetBulb',
    'is_roadside'
]

RANDOM_STATE = 42


def load_data():
    """Load processed data."""
    print("Loading processed data...")
    data = pd.read_csv(os.path.join(DATA_DIR, 'processed_data.csv'), parse_dates=['datetime'])
    print(f"  Shape: {data.shape}")
    print(f"  High-risk rate: {data['target'].mean()*100:.2f}%")
    
    # Sort by datetime to ensure temporal order
    data = data.sort_values('datetime').reset_index(drop=True)
    
    # Fill NaN in feature columns with forward fill then median
    for col in FEATURE_COLS:
        if data[col].isna().any():
            median_val = data[col].median()
            data[col] = data[col].fillna(median_val)
    
    print(f"  NaN remaining in features: {data[FEATURE_COLS].isna().sum().sum()}")
    
    return data


def prepare_train_test(data, test_size=0.2):
    """Split data temporally - last 20% as test set."""
    n = len(data)
    split_idx = int(n * (1 - test_size))
    
    train = data.iloc[:split_idx].copy()
    test = data.iloc[split_idx:].copy()
    
    print(f"\nTemporal split:")
    print(f"  Train: {len(train):,} samples ({train['datetime'].min()} to {train['datetime'].max()})")
    print(f"  Test:  {len(test):,} samples ({test['datetime'].min()} to {test['datetime'].max()})")
    print(f"  Train high-risk rate: {train['target'].mean()*100:.2f}%")
    print(f"  Test high-risk rate:  {test['target'].mean()*100:.2f}%")
    
    return train, test


def train_logistic_regression(X_train, y_train):
    """Train Logistic Regression with class weights."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    model = LogisticRegression(
        class_weight='balanced',
        C=1.0,
        penalty='l2',
        solver='lbfgs',
        max_iter=1000,
        random_state=RANDOM_STATE
    )
    model.fit(X_train_scaled, y_train)
    
    return model, scaler


def train_random_forest(X_train, y_train):
    """Train Random Forest with class weights."""
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=50,
        min_samples_leaf=20,
        class_weight='balanced',
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model, None


def train_xgboost(X_train, y_train):
    """Train XGBoost with scale_pos_weight."""
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos_weight = neg_count / pos_count
    
    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        eval_metric='aucpr',
        early_stopping_rounds=20
    )
    model.fit(X_train, y_train, eval_set=[(X_train, y_train)], verbose=False)
    return model, None


def find_optimal_threshold(y_true, y_proba, min_recall=0.7):
    """Find optimal threshold that maximizes F1 while maintaining minimum recall."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    
    # Filter to thresholds where recall >= min_recall
    valid_mask = recall[:-1] >= min_recall
    
    if valid_mask.sum() == 0:
        # If no threshold meets min_recall, use the one with highest recall
        best_idx = np.argmax(recall[:-1])
        return thresholds[best_idx], precision[best_idx], recall[best_idx]
    
    # Among valid thresholds, find the one with highest F1
    valid_precisions = precision[:-1][valid_mask]
    valid_recalls = recall[:-1][valid_mask]
    valid_thresholds = thresholds[valid_mask]
    
    f1s = 2 * valid_precisions * valid_recalls / (valid_precisions + valid_recalls + 1e-10)
    best_idx = np.argmax(f1s)
    
    return valid_thresholds[best_idx], valid_precisions[best_idx], valid_recalls[best_idx]


def evaluate_model(model, scaler, X_test, y_test, model_name, threshold=0.5):
    """Evaluate a model on the test set."""
    if scaler is not None:
        X_test_scaled = scaler.transform(X_test)
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
    else:
        y_proba = model.predict_proba(X_test)[:, 1]
    
    # Default threshold predictions
    y_pred_default = (y_proba >= 0.5).astype(int)
    
    # Find optimal threshold
    opt_threshold, opt_precision, opt_recall = find_optimal_threshold(y_test, y_proba, min_recall=0.65)
    y_pred_opt = (y_proba >= opt_threshold).astype(int)
    
    # Calculate metrics for both thresholds
    results = {
        'model': model_name,
        'default_threshold': {
            'threshold': 0.5,
            'precision': float(precision_score(y_test, y_pred_default, zero_division=0)),
            'recall': float(recall_score(y_test, y_pred_default, zero_division=0)),
            'f1': float(f1_score(y_test, y_pred_default, zero_division=0)),
            'pr_auc': float(average_precision_score(y_test, y_proba)),
            'confusion_matrix': confusion_matrix(y_test, y_pred_default).tolist(),
        },
        'optimal_threshold': {
            'threshold': float(opt_threshold),
            'precision': float(precision_score(y_test, y_pred_opt, zero_division=0)),
            'recall': float(recall_score(y_test, y_pred_opt, zero_division=0)),
            'f1': float(f1_score(y_test, y_pred_opt, zero_division=0)),
            'pr_auc': float(average_precision_score(y_test, y_proba)),
            'confusion_matrix': confusion_matrix(y_test, y_pred_opt).tolist(),
        },
        'y_proba': y_proba.tolist(),
        'y_test': y_test.tolist(),
    }
    
    return results


def run_cross_validation(data, n_splits=5):
    """Run TimeSeriesSplit cross-validation."""
    print("\n" + "=" * 60)
    print("Running TimeSeriesSplit Cross-Validation")
    print("=" * 60)
    
    X = data[FEATURE_COLS].values
    y = data['target'].values
    
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    cv_results = {
        'logistic_regression': [],
        'random_forest': [],
        'xgboost': []
    }
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        print(f"\n--- Fold {fold+1}/{n_splits} ---")
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        print(f"  Train: {len(X_train):,} (high-risk: {y_train.sum():,}, {y_train.mean()*100:.2f}%)")
        print(f"  Val:   {len(X_val):,} (high-risk: {y_val.sum():,}, {y_val.mean()*100:.2f}%)")
        
        # Logistic Regression
        lr_model, lr_scaler = train_logistic_regression(X_train, y_train)
        lr_results = evaluate_model(lr_model, lr_scaler, X_val, y_val, 'Logistic Regression')
        cv_results['logistic_regression'].append({
            'precision': lr_results['optimal_threshold']['precision'],
            'recall': lr_results['optimal_threshold']['recall'],
            'f1': lr_results['optimal_threshold']['f1'],
            'pr_auc': lr_results['optimal_threshold']['pr_auc'],
        })
        print(f"  LR:   P={lr_results['optimal_threshold']['precision']:.4f}  R={lr_results['optimal_threshold']['recall']:.4f}  F1={lr_results['optimal_threshold']['f1']:.4f}  PR-AUC={lr_results['optimal_threshold']['pr_auc']:.4f}")
        
        # Random Forest
        rf_model, _ = train_random_forest(X_train, y_train)
        rf_results = evaluate_model(rf_model, None, X_val, y_val, 'Random Forest')
        cv_results['random_forest'].append({
            'precision': rf_results['optimal_threshold']['precision'],
            'recall': rf_results['optimal_threshold']['recall'],
            'f1': rf_results['optimal_threshold']['f1'],
            'pr_auc': rf_results['optimal_threshold']['pr_auc'],
        })
        print(f"  RF:   P={rf_results['optimal_threshold']['precision']:.4f}  R={rf_results['optimal_threshold']['recall']:.4f}  F1={rf_results['optimal_threshold']['f1']:.4f}  PR-AUC={rf_results['optimal_threshold']['pr_auc']:.4f}")
        
        # XGBoost
        xgb_model, _ = train_xgboost(X_train, y_train)
        xgb_results = evaluate_model(xgb_model, None, X_val, y_val, 'XGBoost')
        cv_results['xgboost'].append({
            'precision': xgb_results['optimal_threshold']['precision'],
            'recall': xgb_results['optimal_threshold']['recall'],
            'f1': xgb_results['optimal_threshold']['f1'],
            'pr_auc': xgb_results['optimal_threshold']['pr_auc'],
        })
        print(f"  XGB:  P={xgb_results['optimal_threshold']['precision']:.4f}  R={xgb_results['optimal_threshold']['recall']:.4f}  F1={xgb_results['optimal_threshold']['f1']:.4f}  PR-AUC={xgb_results['optimal_threshold']['pr_auc']:.4f}")
    
    # Summarize CV results
    print("\n" + "=" * 60)
    print("Cross-Validation Summary (mean +/- std)")
    print("=" * 60)
    
    cv_summary = {}
    for model_name, folds in cv_results.items():
        metrics = {}
        for metric in ['precision', 'recall', 'f1', 'pr_auc']:
            values = [f[metric] for f in folds]
            metrics[metric] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
            }
        cv_summary[model_name] = metrics
        print(f"\n{model_name}:")
        for metric, vals in metrics.items():
            print(f"  {metric}: {vals['mean']:.4f} +/- {vals['std']:.4f}")
    
    return cv_summary, cv_results


def train_final_models(train, test):
    """Train final models on full training set and evaluate on test set."""
    print("\n" + "=" * 60)
    print("Training Final Models on Full Training Set")
    print("=" * 60)
    
    X_train = train[FEATURE_COLS].values
    y_train = train['target'].values
    X_test = test[FEATURE_COLS].values
    y_test = test['target'].values
    
    print(f"\nTraining set: {len(X_train):,} samples")
    print(f"Test set: {len(X_test):,} samples")
    
    all_results = {}
    trained_models = {}
    
    # Logistic Regression
    print("\nTraining Logistic Regression...")
    lr_model, lr_scaler = train_logistic_regression(X_train, y_train)
    lr_results = evaluate_model(lr_model, lr_scaler, X_test, y_test, 'Logistic Regression')
    all_results['logistic_regression'] = lr_results
    trained_models['logistic_regression'] = (lr_model, lr_scaler)
    
    # Random Forest
    print("Training Random Forest...")
    rf_model, _ = train_random_forest(X_train, y_train)
    rf_results = evaluate_model(rf_model, None, X_test, y_test, 'Random Forest')
    all_results['random_forest'] = rf_results
    trained_models['random_forest'] = (rf_model, None)
    
    # XGBoost
    print("Training XGBoost...")
    xgb_model, _ = train_xgboost(X_train, y_train)
    xgb_results = evaluate_model(xgb_model, None, X_test, y_test, 'XGBoost')
    all_results['xgboost'] = xgb_results
    trained_models['xgboost'] = (xgb_model, None)
    
    # Print results
    print("\n" + "=" * 60)
    print("Test Set Results (Optimal Threshold)")
    print("=" * 60)
    for model_name, results in all_results.items():
        r = results['optimal_threshold']
        print(f"\n{model_name}:")
        print(f"  Threshold: {r['threshold']:.4f}")
        print(f"  Precision: {r['precision']:.4f}")
        print(f"  Recall:    {r['recall']:.4f}")
        print(f"  F1:        {r['f1']:.4f}")
        print(f"  PR-AUC:    {r['pr_auc']:.4f}")
        cm = np.array(r['confusion_matrix'])
        print(f"  Confusion Matrix:")
        print(f"    TN={cm[0][0]:>8,}  FP={cm[0][1]:>8,}")
        print(f"    FN={cm[1][0]:>8,}  TP={cm[1][1]:>8,}")
    
    # Feature importance
    print("\n" + "=" * 60)
    print("Feature Importance")
    print("=" * 60)
    
    feature_importance = {}
    
    # Random Forest feature importance
    rf_importance = trained_models['random_forest'][0].feature_importances_
    rf_imp_df = pd.DataFrame({'feature': FEATURE_COLS, 'importance': rf_importance})
    rf_imp_df = rf_imp_df.sort_values('importance', ascending=False)
    feature_importance['random_forest'] = rf_imp_df.to_dict('records')
    print("\nRandom Forest - Top 10 Features:")
    for _, row in rf_imp_df.head(10).iterrows():
        print(f"  {row['feature']:>25s}: {row['importance']:.4f}")
    
    # XGBoost feature importance
    xgb_importance = trained_models['xgboost'][0].feature_importances_
    xgb_imp_df = pd.DataFrame({'feature': FEATURE_COLS, 'importance': xgb_importance})
    xgb_imp_df = xgb_imp_df.sort_values('importance', ascending=False)
    feature_importance['xgboost'] = xgb_imp_df.to_dict('records')
    print("\nXGBoost - Top 10 Features:")
    for _, row in xgb_imp_df.head(10).iterrows():
        print(f"  {row['feature']:>25s}: {row['importance']:.4f}")
    
    # Logistic Regression coefficients
    lr_model = trained_models['logistic_regression'][0]
    lr_scaler = trained_models['logistic_regression'][1]
    lr_coef = lr_model.coef_[0]
    lr_coef_df = pd.DataFrame({'feature': FEATURE_COLS, 'coefficient': lr_coef})
    lr_coef_df['abs_coefficient'] = lr_coef_df['coefficient'].abs()
    lr_coef_df = lr_coef_df.sort_values('abs_coefficient', ascending=False)
    feature_importance['logistic_regression'] = lr_coef_df.to_dict('records')
    print("\nLogistic Regression - Top 10 |Coefficients|:")
    for _, row in lr_coef_df.head(10).iterrows():
        print(f"  {row['feature']:>25s}: {row['coefficient']:+.4f}")
    
    # Cost-benefit analysis
    print("\n" + "=" * 60)
    print("Cost-Benefit Analysis")
    print("=" * 60)
    print("Assumption: Missing a high-risk alert (FN) is 5x more costly")
    print("            than a false alarm (FP)")
    
    cost_fn = 5
    cost_fp = 1
    
    for model_name, results in all_results.items():
        cm = np.array(results['optimal_threshold']['confusion_matrix'])
        tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
        total_cost = fn * cost_fn + fp * cost_fp
        # Baseline: predict all as non-high-risk
        baseline_cost = (tp + fn) * cost_fn
        cost_reduction = (baseline_cost - total_cost) / baseline_cost * 100
        
        print(f"\n{model_name}:")
        print(f"  Missed alerts (FN): {fn:,}")
        print(f"  False alarms (FP):  {fp:,}")
        print(f"  Total cost:         {total_cost:,}")
        print(f"  Baseline cost:      {baseline_cost:,}")
        print(f"  Cost reduction:     {cost_reduction:.2f}%")
        
        results['cost_analysis'] = {
            'fn': int(fn), 'fp': int(fp),
            'total_cost': int(total_cost),
            'baseline_cost': int(baseline_cost),
            'cost_reduction_pct': float(cost_reduction),
        }
    
    return all_results, trained_models, feature_importance


def main():
    # Load data
    data = load_data()
    
    # Cross-validation
    cv_summary, cv_results = run_cross_validation(data, n_splits=5)
    
    # Train final models
    train, test = prepare_train_test(data, test_size=0.2)
    all_results, trained_models, feature_importance = train_final_models(train, test)
    
    # Save results
    results_output = {
        'cv_summary': cv_summary,
        'cv_results': cv_results,
        'test_results': {k: {kk: vv for kk, vv in v.items() if kk not in ['y_proba', 'y_test']} 
                         for k, v in all_results.items()},
        'feature_importance': feature_importance,
        'feature_cols': FEATURE_COLS,
    }
    
    results_path = os.path.join(OUTPUT_DIR, 'model_results.json')
    with open(results_path, 'w') as f:
        json.dump(results_output, f, indent=2)
    print(f"\nResults saved to: {results_path}")
    
    # Save predictions for plotting
    predictions = {}
    for model_name, results in all_results.items():
        predictions[model_name] = {
            'y_proba': results['y_proba'],
            'y_test': results['y_test'],
        }
    pred_path = os.path.join(OUTPUT_DIR, 'predictions.json')
    with open(pred_path, 'w') as f:
        json.dump(predictions, f)
    print(f"Predictions saved to: {pred_path}")
    
    return all_results, cv_summary, feature_importance


if __name__ == '__main__':
    main()
