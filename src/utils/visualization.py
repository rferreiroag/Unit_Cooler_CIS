"""
Visualization utilities for Unit Cooler Digital Twin EDA
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Optional
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style('whitegrid')
sns.set_palette('Set2')


def plot_missing_values(df: pd.DataFrame, save_path: Optional[str] = None):
    """
    Plot missing values heatmap and bar chart

    Args:
        df: DataFrame to analyze
        save_path: Path to save figure (optional)
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Missing values percentage
    missing_pct = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
    missing_pct = missing_pct[missing_pct > 0]

    if len(missing_pct) > 0:
        # Bar chart
        axes[0].barh(range(len(missing_pct)), missing_pct.values, color='coral')
        axes[0].set_yticks(range(len(missing_pct)))
        axes[0].set_yticklabels(missing_pct.index, fontsize=8)
        axes[0].set_xlabel('Missing Values (%)', fontsize=10)
        axes[0].set_title('Missing Values by Variable', fontsize=12, fontweight='bold')
        axes[0].grid(axis='x', alpha=0.3)

        # Add percentage labels
        for i, v in enumerate(missing_pct.values):
            axes[0].text(v + 1, i, f'{v:.1f}%', va='center', fontsize=8)

        # Heatmap sample (first 1000 rows)
        sample_size = min(1000, len(df))
        axes[1].imshow(df.head(sample_size).isnull().T, cmap='RdYlGn_r', aspect='auto', interpolation='nearest')
        axes[1].set_xlabel('Row Index (first 1000)', fontsize=10)
        axes[1].set_ylabel('Variables', fontsize=10)
        axes[1].set_title('Missing Values Pattern (Sample)', fontsize=12, fontweight='bold')
        axes[1].set_yticks([])
    else:
        axes[0].text(0.5, 0.5, 'No missing values', ha='center', va='center', fontsize=14)
        axes[0].axis('off')
        axes[1].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")

    plt.close()


def plot_distributions(df: pd.DataFrame, variables: List[str], save_path: Optional[str] = None):
    """
    Plot distributions for specified variables

    Args:
        df: DataFrame to analyze
        variables: List of variable names
        save_path: Path to save figure (optional)
    """
    available_vars = [v for v in variables if v in df.columns]
    n_vars = len(available_vars)

    if n_vars == 0:
        print("No variables available to plot")
        return

    n_cols = 3
    n_rows = (n_vars + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
    axes = axes.flatten() if n_vars > 1 else [axes]

    for idx, var in enumerate(available_vars):
        data = df[var].dropna()

        if len(data) > 0:
            axes[idx].hist(data, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
            axes[idx].set_xlabel(var, fontsize=10)
            axes[idx].set_ylabel('Frequency', fontsize=10)
            axes[idx].set_title(f'{var} Distribution', fontsize=11, fontweight='bold')
            axes[idx].grid(axis='y', alpha=0.3)

            # Add statistics
            stats_text = f'Mean: {data.mean():.2f}\nStd: {data.std():.2f}\nMin: {data.min():.2f}\nMax: {data.max():.2f}'
            axes[idx].text(0.98, 0.97, stats_text, transform=axes[idx].transAxes,
                          fontsize=8, verticalalignment='top', horizontalalignment='right',
                          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        else:
            axes[idx].text(0.5, 0.5, 'No data', ha='center', va='center', fontsize=12)
            axes[idx].axis('off')

    # Hide unused subplots
    for idx in range(n_vars, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")

    plt.close()


def plot_correlation_heatmap(df: pd.DataFrame, variables: Optional[List[str]] = None,
                            save_path: Optional[str] = None):
    """
    Plot correlation heatmap

    Args:
        df: DataFrame to analyze
        variables: List of variables to include (None = all numeric)
        save_path: Path to save figure (optional)
    """
    if variables is None:
        variables = df.select_dtypes(include=[np.number]).columns.tolist()

    available_vars = [v for v in variables if v in df.columns]

    if len(available_vars) < 2:
        print("Not enough variables for correlation analysis")
        return

    # Calculate correlation
    corr_data = df[available_vars].corr()

    # Plot
    fig, ax = plt.subplots(figsize=(14, 12))

    mask = np.triu(np.ones_like(corr_data, dtype=bool))

    sns.heatmap(corr_data, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, vmin=-1, vmax=1, square=True, linewidths=0.5,
                cbar_kws={"shrink": 0.8}, ax=ax, annot_kws={'fontsize': 7})

    ax.set_title('Correlation Matrix', fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")

    plt.close()


def plot_time_series(df: pd.DataFrame, variables: List[str], sample_size: int = 5000,
                     save_path: Optional[str] = None):
    """
    Plot time series for specified variables

    Args:
        df: DataFrame to analyze
        variables: List of variable names
        sample_size: Number of points to plot
        save_path: Path to save figure (optional)
    """
    available_vars = [v for v in variables if v in df.columns]
    n_vars = len(available_vars)

    if n_vars == 0:
        print("No variables available to plot")
        return

    # Sample data if too large
    if len(df) > sample_size:
        df_plot = df.sample(n=sample_size, random_state=42).sort_index()
    else:
        df_plot = df

    fig, axes = plt.subplots(n_vars, 1, figsize=(14, 3*n_vars))
    if n_vars == 1:
        axes = [axes]

    for idx, var in enumerate(available_vars):
        data = df_plot[var].dropna()

        if len(data) > 0:
            axes[idx].plot(data.index, data.values, linewidth=0.8, alpha=0.7, color='steelblue')
            axes[idx].set_ylabel(var, fontsize=10, fontweight='bold')
            axes[idx].set_title(f'{var} Time Series', fontsize=11)
            axes[idx].grid(True, alpha=0.3)

            # Add horizontal line at mean
            mean_val = data.mean()
            axes[idx].axhline(y=mean_val, color='red', linestyle='--', linewidth=1, alpha=0.5, label=f'Mean: {mean_val:.2f}')
            axes[idx].legend(loc='upper right', fontsize=8)
        else:
            axes[idx].text(0.5, 0.5, 'No data', ha='center', va='center', fontsize=12)

    axes[-1].set_xlabel('Sample Index', fontsize=10)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")

    plt.close()


def plot_boxplots(df: pd.DataFrame, variables: List[str], save_path: Optional[str] = None):
    """
    Plot boxplots for specified variables

    Args:
        df: DataFrame to analyze
        variables: List of variable names
        save_path: Path to save figure (optional)
    """
    available_vars = [v for v in variables if v in df.columns]
    n_vars = len(available_vars)

    if n_vars == 0:
        print("No variables available to plot")
        return

    n_cols = 3
    n_rows = (n_vars + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
    axes = axes.flatten() if n_vars > 1 else [axes]

    for idx, var in enumerate(available_vars):
        data = df[var].dropna()

        if len(data) > 0:
            bp = axes[idx].boxplot(data, vert=True, patch_artist=True,
                                  boxprops=dict(facecolor='lightblue', alpha=0.7),
                                  medianprops=dict(color='red', linewidth=2))
            axes[idx].set_ylabel(var, fontsize=10)
            axes[idx].set_title(f'{var} Boxplot', fontsize=11, fontweight='bold')
            axes[idx].grid(axis='y', alpha=0.3)
            axes[idx].set_xticklabels([''])

            # Add outlier count
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((data < (Q1 - 1.5*IQR)) | (data > (Q3 + 1.5*IQR))).sum()
            outlier_pct = outliers / len(data) * 100

            axes[idx].text(0.98, 0.97, f'Outliers: {outliers}\n({outlier_pct:.1f}%)',
                          transform=axes[idx].transAxes, fontsize=8,
                          verticalalignment='top', horizontalalignment='right',
                          bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
        else:
            axes[idx].text(0.5, 0.5, 'No data', ha='center', va='center', fontsize=12)
            axes[idx].axis('off')

    # Hide unused subplots
    for idx in range(n_vars, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")

    plt.close()


def plot_target_correlations(df: pd.DataFrame, target_vars: List[str], top_n: int = 15,
                            save_path: Optional[str] = None):
    """
    Plot correlations with target variables

    Args:
        df: DataFrame to analyze
        target_vars: List of target variable names
        top_n: Number of top correlations to show
        save_path: Path to save figure (optional)
    """
    available_targets = [t for t in target_vars if t in df.columns]

    if len(available_targets) == 0:
        print("No target variables available")
        return

    n_targets = len(available_targets)
    fig, axes = plt.subplots(1, n_targets, figsize=(6*n_targets, 8))

    if n_targets == 1:
        axes = [axes]

    for idx, target in enumerate(available_targets):
        corr = df.corr()[target].drop(target).abs().sort_values(ascending=False).head(top_n)

        axes[idx].barh(range(len(corr)), corr.values, color='steelblue', alpha=0.7)
        axes[idx].set_yticks(range(len(corr)))
        axes[idx].set_yticklabels(corr.index, fontsize=9)
        axes[idx].set_xlabel('|Correlation|', fontsize=10)
        axes[idx].set_title(f'Top {top_n} Correlations with {target}', fontsize=11, fontweight='bold')
        axes[idx].grid(axis='x', alpha=0.3)
        axes[idx].invert_yaxis()

        # Add correlation values
        for i, v in enumerate(corr.values):
            axes[idx].text(v + 0.01, i, f'{v:.3f}', va='center', fontsize=8)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")

    plt.close()


def generate_all_eda_plots(df: pd.DataFrame, output_dir: str = 'plots'):
    """
    Generate all EDA plots

    Args:
        df: DataFrame to analyze
        output_dir: Directory to save plots
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("\n=== GENERATING EDA VISUALIZATIONS ===\n")

    # 1. Missing values
    print("1. Missing values plot...")
    plot_missing_values(df, save_path=output_path / 'missing_values.png')

    # 2. Key variables distributions
    print("2. Distribution plots...")
    key_vars = ['UCWIT', 'UCWOT', 'UCAIT', 'UCAOT', 'UCWF', 'UCAF', 'UCAIH', 'AMBT', 'UCTSP']
    plot_distributions(df, key_vars, save_path=output_path / 'distributions_key_variables.png')

    # 3. Correlation heatmap (key variables only to avoid clutter)
    print("3. Correlation heatmap...")
    plot_correlation_heatmap(df, variables=key_vars + ['CPSP', 'CPPR', 'CPDP'],
                            save_path=output_path / 'correlation_heatmap.png')

    # 4. Time series
    print("4. Time series plots...")
    plot_time_series(df, key_vars, sample_size=5000,
                    save_path=output_path / 'time_series_key_variables.png')

    # 5. Boxplots
    print("5. Boxplot analysis...")
    plot_boxplots(df, key_vars, save_path=output_path / 'boxplots_key_variables.png')

    # 6. Target correlations
    print("6. Target variable correlations...")
    target_vars = ['UCAOT', 'UCWOT', 'UCAF']
    plot_target_correlations(df, target_vars, top_n=15,
                            save_path=output_path / 'target_correlations.png')

    print("\n✓ All visualizations generated successfully!")
    print(f"   Plots saved to: {output_path.absolute()}")


if __name__ == "__main__":
    pass
