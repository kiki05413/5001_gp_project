"""
DSC5001 Project: Hong Kong AQHI High-Risk Hour Early Warning
=============================================================
Step 4: Generate all visualizations and analysis plots

Plots:
  1. AQHI distribution and class imbalance
  2. PR curves for all three models
  3. Confusion matrices (optimal threshold)
  4. CV metrics comparison (boxplot)
  5. Feature importance (RF & XGBoost)
  6. LR coefficients
  7. Cost-benefit analysis
  8. Hourly AQHI pattern by station type
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
import seaborn as sns

# Style settings
sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['axes.labelsize'] = 11

# Colors
COLORS = {
    'logistic_regression': '#2196F3',   # Blue
    'random_forest': '#4CAF50',          # Green
    'xgboost': '#FF9800',                # Orange
}
MODEL_LABELS = {
    'logistic_regression': 'Logistic Regression',
    'random_forest': 'Random Forest',
    'xgboost': 'XGBoost',
}

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'output'))
FIGURES_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'figures'))
os.makedirs(FIGURES_DIR, exist_ok=True)


def load_results():
    with open(os.path.join(OUTPUT_DIR, 'model_results.json'), 'r') as f:
        return json.load(f)


def load_data():
    return pd.read_csv(os.path.join(DATA_DIR, 'processed_data.csv'), parse_dates=['datetime'])


# ============================================================
# Plot 1: AQHI Distribution & Class Imbalance
# ============================================================
def plot_aqhi_distribution(data):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: AQHI value distribution
    ax = axes[0]
    counts = data['AQHI'].value_counts().sort_index()
    colors = ['#66BB6A' if v < 7 else '#EF5350' for v in counts.index]
    bars = ax.bar(counts.index, counts.values, color=colors, edgecolor='white', linewidth=0.5)
    ax.axvline(x=6.5, color='red', linestyle='--', linewidth=1.5, label='Threshold (AQHI=7)')
    ax.set_xlabel('AQHI Value')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Hourly AQHI Values')
    ax.legend()
    
    # Add percentage labels on bars
    total = counts.sum()
    for bar, v in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + total*0.005,
                f'{v/total*100:.1f}%', ha='center', va='bottom', fontsize=7)
    
    # Right: Target class distribution
    ax = axes[1]
    target_counts = data['target'].value_counts()
    labels = ['Non-High-Risk\n(AQHI < 7)', 'High-Risk\n(AQHI >= 7)']
    colors_pie = ['#66BB6A', '#EF5350']
    wedges, texts, autotexts = ax.pie(
        target_counts.values, labels=labels, colors=colors_pie,
        autopct='%1.2f%%', startangle=90, textprops={'fontsize': 11}
    )
    ax.set_title('Target Class Distribution\n(Class Imbalance Visualization)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, '01_aqhi_distribution.png'), bbox_inches='tight')
    plt.close()
    print("  Saved: 01_aqhi_distribution.png")


# ============================================================
# Plot 2: PR Curves
# ============================================================
def plot_pr_curves(results):
    # Load predictions
    with open(os.path.join(OUTPUT_DIR, 'predictions.json'), 'r') as f:
        predictions = json.load(f)
    
    from sklearn.metrics import precision_recall_curve, average_precision_score
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    for model_key in ['logistic_regression', 'random_forest', 'xgboost']:
        y_test = np.array(predictions[model_key]['y_test'])
        y_proba = np.array(predictions[model_key]['y_proba'])
        
        precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
        pr_auc = average_precision_score(y_test, y_proba)
        
        ax.plot(recall, precision, color=COLORS[model_key], linewidth=2,
                label=f'{MODEL_LABELS[model_key]} (PR-AUC={pr_auc:.4f})')
    
    # Baseline (random classifier)
    positive_rate = np.mean(predictions['logistic_regression']['y_test'])
    ax.axhline(y=positive_rate, color='gray', linestyle='--', linewidth=1,
               label=f'Random Baseline ({positive_rate:.4f})')
    
    ax.set_xlabel('Recall (Sensitivity)')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curves (Test Set)')
    ax.legend(loc='upper right', fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, '02_pr_curves.png'), bbox_inches='tight')
    plt.close()
    print("  Saved: 02_pr_curves.png")


# ============================================================
# Plot 3: Confusion Matrices
# ============================================================
def plot_confusion_matrices(results):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, model_key in enumerate(['logistic_regression', 'random_forest', 'xgboost']):
        ax = axes[idx]
        cm = np.array(results['test_results'][model_key]['optimal_threshold']['confusion_matrix'])
        
        # Normalize for display
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Plot as heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Non-High-Risk', 'High-Risk'],
                    yticklabels=['Non-High-Risk', 'High-Risk'],
                    cbar=False)
        
        # Add normalized values as secondary annotation
        for i in range(2):
            for j in range(2):
                ax.text(j + 0.5, i + 0.65, f'({cm_norm[i][j]*100:.1f}%)',
                        ha='center', va='center', fontsize=8, color='gray')
        
        threshold = results['test_results'][model_key]['optimal_threshold']['threshold']
        ax.set_title(f'{MODEL_LABELS[model_key]}\n(threshold={threshold:.3f})')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
    
    plt.suptitle('Confusion Matrices (Optimal Threshold)', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, '03_confusion_matrices.png'), bbox_inches='tight')
    plt.close()
    print("  Saved: 03_confusion_matrices.png")


# ============================================================
# Plot 4: CV Metrics Comparison (Boxplot)
# ============================================================
def plot_cv_comparison(results):
    cv_data = results['cv_results']
    
    metrics = ['precision', 'recall', 'f1', 'pr_auc']
    metric_labels = ['Precision', 'Recall', 'F1 Score', 'PR-AUC']
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
        ax = axes[idx // 2][idx % 2]
        
        plot_data = []
        labels = []
        colors = []
        
        for model_key in ['logistic_regression', 'random_forest', 'xgboost']:
            values = [fold[metric] for fold in cv_data[model_key]]
            plot_data.append(values)
            labels.append(MODEL_LABELS[model_key])
            colors.append(COLORS[model_key])
        
        bp = ax.boxplot(plot_data, tick_labels=labels, patch_artist=True, widths=0.6)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        # Overlay individual points
        for i, (values, color) in enumerate(zip(plot_data, colors)):
            x = np.random.normal(i + 1, 0.04, size=len(values))
            ax.scatter(x, values, color=color, alpha=0.8, s=30, zorder=3)
        
        ax.set_ylabel(label)
        ax.set_title(f'{label} (5-Fold TimeSeriesSplit)')
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Rotate x labels
        ax.tick_params(axis='x', rotation=15)
    
    plt.suptitle('Cross-Validation Metrics Comparison', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, '04_cv_comparison.png'), bbox_inches='tight')
    plt.close()
    print("  Saved: 04_cv_comparison.png")


# ============================================================
# Plot 5: Feature Importance (RF & XGBoost)
# ============================================================
def plot_feature_importance(results):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    for idx, model_key in enumerate(['random_forest', 'xgboost']):
        ax = axes[idx]
        imp_data = results['feature_importance'][model_key]
        
        # Take top 12 features
        df = pd.DataFrame(imp_data[:12])
        
        # Clean feature names for display
        df['feature'] = df['feature'].str.replace('_', ' ').str.title()
        
        bars = ax.barh(df['feature'][::-1], df['importance'][::-1],
                       color=COLORS[model_key], alpha=0.8, edgecolor='white')
        ax.set_xlabel('Feature Importance')
        ax.set_title(f'{MODEL_LABELS[model_key]} - Top 12 Features')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for bar, val in zip(bars, df['importance'][::-1]):
            ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                    f'{val:.4f}', va='center', fontsize=8)
    
    plt.suptitle('Feature Importance Comparison', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, '05_feature_importance.png'), bbox_inches='tight')
    plt.close()
    print("  Saved: 05_feature_importance.png")


# ============================================================
# Plot 6: Logistic Regression Coefficients
# ============================================================
def plot_lr_coefficients(results):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    coef_data = results['feature_importance']['logistic_regression']
    df = pd.DataFrame(coef_data[:15])
    df['feature'] = df['feature'].str.replace('_', ' ').str.title()
    
    colors = ['#EF5350' if c > 0 else '#42A5F5' for c in df['coefficient']]
    
    bars = ax.barh(df['feature'][::-1], df['coefficient'][::-1],
                   color=colors[::-1], alpha=0.8, edgecolor='white')
    ax.set_xlabel('Coefficient Value')
    ax.set_title('Logistic Regression - Top 15 Features by |Coefficient|')
    ax.axvline(x=0, color='black', linewidth=0.8)
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for bar, val in zip(bars, df['coefficient'][::-1]):
        offset = 0.03 if val > 0 else -0.03
        ha = 'left' if val > 0 else 'right'
        ax.text(bar.get_width() + offset, bar.get_y() + bar.get_height()/2,
                f'{val:+.3f}', va='center', ha=ha, fontsize=8)
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#EF5350', alpha=0.8, label='Positive (increases risk)'),
                       Patch(facecolor='#42A5F5', alpha=0.8, label='Negative (decreases risk)')]
    ax.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, '06_lr_coefficients.png'), bbox_inches='tight')
    plt.close()
    print("  Saved: 06_lr_coefficients.png")


# ============================================================
# Plot 7: Cost-Benefit Analysis
# ============================================================
def plot_cost_benefit(results):
    test_results = results['test_results']
    
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    
    # Left: FN vs FP comparison
    ax = axes[0]
    models = []
    fn_values = []
    fp_values = []
    tp_values = []
    
    for model_key in ['logistic_regression', 'random_forest', 'xgboost']:
        models.append(MODEL_LABELS[model_key])
        cost = test_results[model_key]['cost_analysis']
        fn_values.append(cost['fn'])
        fp_values.append(cost['fp'])
        tp_values.append(test_results[model_key]['optimal_threshold']['confusion_matrix'][1][1])
    
    x = np.arange(len(models))
    width = 0.25
    
    bars1 = ax.bar(x - width, fn_values, width, label='Missed Alerts (FN)', color='#EF5350', alpha=0.8)
    bars2 = ax.bar(x, fp_values, width, label='False Alarms (FP)', color='#FFA726', alpha=0.8)
    bars3 = ax.bar(x + width, tp_values, width, label='Correct Alerts (TP)', color='#66BB6A', alpha=0.8)
    
    ax.set_xlabel('Model')
    ax.set_ylabel('Count')
    ax.set_title('Alert Outcomes on Test Set')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                    f'{int(bar.get_height())}', ha='center', va='bottom', fontsize=8)
    
    # Right: Cost reduction
    ax = axes[1]
    cost_reductions = []
    total_costs = []
    baseline_costs = []
    
    for model_key in ['logistic_regression', 'random_forest', 'xgboost']:
        cost = test_results[model_key]['cost_analysis']
        cost_reductions.append(cost['cost_reduction_pct'])
        total_costs.append(cost['total_cost'])
        baseline_costs.append(cost['baseline_cost'])
    
    x = np.arange(len(models))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, baseline_costs, width, label='Baseline Cost (No Model)', color='#BDBDBD', alpha=0.8)
    bars2 = ax.bar(x + width/2, total_costs, width, label='Model Cost', color='#42A5F5', alpha=0.8)
    
    ax.set_xlabel('Model')
    ax.set_ylabel('Total Cost (FN=5, FP=1)')
    ax.set_title('Cost-Benefit Analysis\n(Missing high-risk alert = 5x cost of false alarm)')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add cost reduction percentage annotations
    for i, reduction in enumerate(cost_reductions):
        ax.annotate(f'-{reduction:.1f}%', xy=(i, max(baseline_costs[i], total_costs[i]) + 100),
                    ha='center', fontsize=10, fontweight='bold', color='#2E7D32')
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, '07_cost_benefit.png'), bbox_inches='tight')
    plt.close()
    print("  Saved: 07_cost_benefit.png")


# ============================================================
# Plot 8: Hourly AQHI Pattern by Station Type
# ============================================================
def plot_hourly_pattern(data):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Average AQHI by hour and station type
    ax = axes[0]
    hourly_avg = data.groupby(['hour', 'is_roadside'])['AQHI'].mean().unstack()
    
    ax.plot(hourly_avg.index, hourly_avg[1], color='#EF5350', linewidth=2, marker='o', markersize=4, label='Roadside')
    ax.plot(hourly_avg.index, hourly_avg[0], color='#42A5F5', linewidth=2, marker='s', markersize=4, label='General')
    ax.axhline(y=7, color='red', linestyle='--', linewidth=1, alpha=0.5, label='High-Risk Threshold (7)')
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Average AQHI')
    ax.set_title('Average AQHI by Hour and Station Type')
    ax.set_xticks(range(0, 24, 2))
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Right: High-risk rate by hour
    ax = axes[1]
    hourly_risk = data.groupby(['hour', 'is_roadside'])['target'].mean().unstack() * 100
    
    ax.plot(hourly_risk.index, hourly_risk[1], color='#EF5350', linewidth=2, marker='o', markersize=4, label='Roadside')
    ax.plot(hourly_risk.index, hourly_risk[0], color='#42A5F5', linewidth=2, marker='s', markersize=4, label='General')
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('High-Risk Rate (%)')
    ax.set_title('High-Risk (AQHI>=7) Rate by Hour and Station Type')
    ax.set_xticks(range(0, 24, 2))
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, '08_hourly_pattern.png'), bbox_inches='tight')
    plt.close()
    print("  Saved: 08_hourly_pattern.png")


# ============================================================
# Plot 9: Monthly High-Risk Pattern
# ============================================================
def plot_monthly_pattern(data):
    fig, ax = plt.subplots(figsize=(10, 5))
    
    monthly_risk = data.groupby(['month', 'is_roadside'])['target'].mean().unstack() * 100
    monthly_count = data.groupby('month')['target'].sum()
    
    ax.bar(monthly_risk.index - 0.15, monthly_risk[0], width=0.3, color='#42A5F5', alpha=0.8, label='General Stations')
    ax.bar(monthly_risk.index + 0.15, monthly_risk[1], width=0.3, color='#EF5350', alpha=0.8, label='Roadside Stations')
    
    ax.set_xlabel('Month')
    ax.set_ylabel('High-Risk Rate (%)')
    ax.set_title('Monthly High-Risk (AQHI>=7) Rate by Station Type')
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, '09_monthly_pattern.png'), bbox_inches='tight')
    plt.close()
    print("  Saved: 09_monthly_pattern.png")


# ============================================================
# Plot 10: Threshold Sensitivity Analysis
# ============================================================
def plot_threshold_sensitivity(results):
    with open(os.path.join(OUTPUT_DIR, 'predictions.json'), 'r') as f:
        predictions = json.load(f)
    
    from sklearn.metrics import precision_recall_curve
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, model_key in enumerate(['logistic_regression', 'random_forest', 'xgboost']):
        ax = axes[idx]
        y_test = np.array(predictions[model_key]['y_test'])
        y_proba = np.array(predictions[model_key]['y_proba'])
        
        precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
        
        # Plot precision and recall vs threshold
        ax.plot(thresholds, precision[:-1], color='#2196F3', linewidth=2, label='Precision')
        ax.plot(thresholds, recall[:-1], color='#EF5350', linewidth=2, label='Recall')
        
        # Plot F1
        f1 = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-10)
        ax.plot(thresholds, f1, color='#4CAF50', linewidth=2, linestyle='--', label='F1')
        
        # Mark optimal threshold
        opt_threshold = results['test_results'][model_key]['optimal_threshold']['threshold']
        ax.axvline(x=opt_threshold, color='black', linestyle=':', linewidth=1.5, label=f'Optimal ({opt_threshold:.3f})')
        
        ax.set_xlabel('Threshold')
        ax.set_ylabel('Score')
        ax.set_title(MODEL_LABELS[model_key])
        ax.legend(fontsize=8)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Threshold Sensitivity Analysis', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, '10_threshold_sensitivity.png'), bbox_inches='tight')
    plt.close()
    print("  Saved: 10_threshold_sensitivity.png")


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 60)
    print("Generating Visualizations")
    print("=" * 60)
    
    results = load_results()
    data = load_data()
    
    print("\nGenerating plots...")
    plot_aqhi_distribution(data)
    plot_pr_curves(results)
    plot_confusion_matrices(results)
    plot_cv_comparison(results)
    plot_feature_importance(results)
    plot_lr_coefficients(results)
    plot_cost_benefit(results)
    plot_hourly_pattern(data)
    plot_monthly_pattern(data)
    plot_threshold_sensitivity(results)
    
    print(f"\nAll figures saved to: {FIGURES_DIR}")
    print(f"Total: 10 figures")


if __name__ == '__main__':
    main()
