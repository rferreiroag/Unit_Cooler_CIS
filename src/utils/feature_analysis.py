"""
Feature importance analysis for Unit Cooler models
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 8)


def plot_feature_importance(model, top_n: int = 20, save_path: Optional[str] = None):
    """
    Plot feature importance for tree-based models

    Args:
        model: Trained model with feature_importance method
        top_n: Number of top features to show
        save_path: Path to save figure (optional)
    """
    importance_df = model.get_feature_importance()

    if importance_df is None:
        print("Feature importance not available for this model")
        return None

    # Get top features by mean importance
    top_features = importance_df.nlargest(top_n, 'mean')

    # Plot
    fig, axes = plt.subplots(1, len(model.target_names) + 1, figsize=(18, 8))

    # Plot for each target
    for idx, target in enumerate(model.target_names):
        if target in top_features.columns:
            top_by_target = importance_df.nlargest(top_n, target)

            axes[idx].barh(range(len(top_by_target)), top_by_target[target].values, color='steelblue', alpha=0.7)
            axes[idx].set_yticks(range(len(top_by_target)))
            axes[idx].set_yticklabels(top_by_target.index, fontsize=9)
            axes[idx].set_xlabel('Importance', fontsize=10)
            axes[idx].set_title(f'{target}\nTop {top_n} Features', fontsize=11, fontweight='bold')
            axes[idx].invert_yaxis()
            axes[idx].grid(axis='x', alpha=0.3)

    # Plot mean importance across all targets
    axes[-1].barh(range(len(top_features)), top_features['mean'].values, color='coral', alpha=0.7)
    axes[-1].set_yticks(range(len(top_features)))
    axes[-1].set_yticklabels(top_features.index, fontsize=9)
    axes[-1].set_xlabel('Mean Importance', fontsize=10)
    axes[-1].set_title(f'Mean Across All Targets\nTop {top_n} Features', fontsize=11, fontweight='bold')
    axes[-1].invert_yaxis()
    axes[-1].grid(axis='x', alpha=0.3)

    plt.suptitle(f'{model.model_type.upper()} Feature Importance Analysis', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")

    plt.close()

    return top_features


def compare_feature_importance_across_models(models: Dict, top_n: int = 15,
                                             save_path: Optional[str] = None):
    """
    Compare feature importance across different models

    Args:
        models: Dictionary of {model_name: model_object}
        top_n: Number of top features to show
        save_path: Path to save figure (optional)
    """
    print("\n" + "="*80)
    print(" FEATURE IMPORTANCE COMPARISON")
    print("="*80)

    # Collect importance from all models
    all_importance = {}

    for model_name, model in models.items():
        if hasattr(model, 'get_feature_importance'):
            importance_df = model.get_feature_importance()
            if importance_df is not None:
                all_importance[model_name] = importance_df['mean']

    if not all_importance:
        print("No feature importance available from models")
        return None

    # Combine into single DataFrame
    importance_combined = pd.DataFrame(all_importance)

    # Get top features by average importance across all models
    importance_combined['avg_all_models'] = importance_combined.mean(axis=1)
    top_features = importance_combined.nlargest(top_n, 'avg_all_models')

    print(f"\nTop {top_n} features (averaged across models):")
    for i, (feature, row) in enumerate(top_features.iterrows(), 1):
        avg_importance = row['avg_all_models']
        print(f"  {i:2d}. {feature:30s}: {avg_importance:.4f}")

    # Plot comparison
    fig, ax = plt.subplots(figsize=(14, 10))

    # Prepare data for grouped bar chart
    x = np.arange(len(top_features))
    width = 0.35
    models_list = list(all_importance.keys())

    for i, model_name in enumerate(models_list):
        offset = width * (i - len(models_list)/2 + 0.5)
        values = top_features[model_name].values
        ax.barh(x + offset, values, width, label=model_name, alpha=0.7)

    ax.set_yticks(x)
    ax.set_yticklabels(top_features.index, fontsize=9)
    ax.set_xlabel('Feature Importance', fontsize=11)
    ax.set_title(f'Top {top_n} Features - Model Comparison', fontsize=13, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Saved comparison plot: {save_path}")

    plt.close()

    # Save to CSV
    if save_path:
        csv_path = Path(save_path).parent / 'feature_importance_comparison.csv'
        top_features.to_csv(csv_path)
        print(f"✓ Saved CSV: {csv_path}")

    return top_features


def analyze_feature_groups(importance_df: pd.DataFrame, feature_groups: Dict[str, List[str]]):
    """
    Analyze importance by feature groups

    Args:
        importance_df: DataFrame with feature importance
        feature_groups: Dictionary mapping group names to feature lists

    Returns:
        DataFrame with group-level importance
    """
    print("\n" + "="*80)
    print(" FEATURE GROUP ANALYSIS")
    print("="*80)

    group_importance = {}

    for group_name, features in feature_groups.items():
        # Find features in this group that are in importance_df
        group_features = [f for f in features if f in importance_df.index]

        if group_features:
            # Calculate mean importance for this group
            mean_importance = importance_df.loc[group_features, 'mean'].mean()
            sum_importance = importance_df.loc[group_features, 'mean'].sum()
            n_features = len(group_features)

            group_importance[group_name] = {
                'mean_importance': mean_importance,
                'sum_importance': sum_importance,
                'n_features': n_features
            }

    # Convert to DataFrame
    group_df = pd.DataFrame(group_importance).T
    group_df = group_df.sort_values('sum_importance', ascending=False)

    print("\nFeature Group Importance:")
    print(group_df.to_string())

    return group_df


def plot_predictions_vs_actual(model, X_test: np.ndarray, y_test: np.ndarray,
                               target_names: List[str], save_path: Optional[str] = None):
    """
    Plot predictions vs actual values for each target

    Args:
        model: Trained model
        X_test: Test features
        y_test: Test targets
        target_names: Target variable names
        save_path: Path to save figure (optional)
    """
    y_pred = model.predict(X_test)

    n_targets = len(target_names)
    fig, axes = plt.subplots(1, n_targets, figsize=(6*n_targets, 5))

    if n_targets == 1:
        axes = [axes]

    for idx, target in enumerate(target_names):
        y_true = y_test[:, idx] if len(y_test.shape) > 1 else y_test
        y_pred_i = y_pred[:, idx] if len(y_pred.shape) > 1 else y_pred

        # Scatter plot
        axes[idx].scatter(y_true, y_pred_i, alpha=0.3, s=10, color='steelblue')

        # Perfect prediction line
        min_val = min(y_true.min(), y_pred_i.min())
        max_val = max(y_true.max(), y_pred_i.max())
        axes[idx].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')

        # Calculate R²
        from sklearn.metrics import r2_score
        r2 = r2_score(y_true, y_pred_i)

        axes[idx].set_xlabel('Actual', fontsize=10)
        axes[idx].set_ylabel('Predicted', fontsize=10)
        axes[idx].set_title(f'{target}\nR² = {r2:.4f}', fontsize=11, fontweight='bold')
        axes[idx].legend(fontsize=9)
        axes[idx].grid(True, alpha=0.3)

    plt.suptitle(f'{model.model_type.upper()} - Predictions vs Actual', fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")

    plt.close()


def plot_residuals(model, X_test: np.ndarray, y_test: np.ndarray,
                   target_names: List[str], save_path: Optional[str] = None):
    """
    Plot residuals analysis for each target

    Args:
        model: Trained model
        X_test: Test features
        y_test: Test targets
        target_names: Target variable names
        save_path: Path to save figure (optional)
    """
    y_pred = model.predict(X_test)

    n_targets = len(target_names)
    fig, axes = plt.subplots(2, n_targets, figsize=(6*n_targets, 10))

    for idx, target in enumerate(target_names):
        y_true = y_test[:, idx] if len(y_test.shape) > 1 else y_test
        y_pred_i = y_pred[:, idx] if len(y_pred.shape) > 1 else y_pred

        residuals = y_true - y_pred_i

        # Residuals vs predicted
        axes[0, idx].scatter(y_pred_i, residuals, alpha=0.3, s=10, color='steelblue')
        axes[0, idx].axhline(y=0, color='r', linestyle='--', linewidth=2)
        axes[0, idx].set_xlabel('Predicted', fontsize=10)
        axes[0, idx].set_ylabel('Residuals', fontsize=10)
        axes[0, idx].set_title(f'{target}\nResiduals vs Predicted', fontsize=11, fontweight='bold')
        axes[0, idx].grid(True, alpha=0.3)

        # Residuals histogram
        axes[1, idx].hist(residuals, bins=50, edgecolor='black', alpha=0.7, color='coral')
        axes[1, idx].axvline(x=0, color='r', linestyle='--', linewidth=2)
        axes[1, idx].set_xlabel('Residuals', fontsize=10)
        axes[1, idx].set_ylabel('Frequency', fontsize=10)
        axes[1, idx].set_title(f'{target}\nResidual Distribution', fontsize=11, fontweight='bold')
        axes[1, idx].grid(True, alpha=0.3)

        # Add statistics
        stats_text = f'Mean: {residuals.mean():.4f}\nStd: {residuals.std():.4f}'
        axes[1, idx].text(0.02, 0.98, stats_text, transform=axes[1, idx].transAxes,
                         fontsize=9, verticalalignment='top',
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle(f'{model.model_type.upper()} - Residual Analysis', fontsize=13, fontweight='bold', y=1.00)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")

    plt.close()


if __name__ == "__main__":
    # Test feature analysis
    pass
