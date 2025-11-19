"""
Exploratory Data Analysis utilities for Unit Cooler Digital Twin
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


class EDAAnalyzer:
    """Comprehensive EDA for Unit Cooler data"""

    def __init__(self, df: pd.DataFrame):
        """
        Initialize EDA Analyzer

        Args:
            df: DataFrame to analyze
        """
        self.df = df
        self.numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    def generate_summary_statistics(self) -> pd.DataFrame:
        """Generate comprehensive summary statistics"""
        stats = self.df[self.numeric_cols].describe()

        # Add additional statistics
        stats.loc['skew'] = self.df[self.numeric_cols].skew()
        stats.loc['kurtosis'] = self.df[self.numeric_cols].kurtosis()
        stats.loc['missing'] = self.df[self.numeric_cols].isnull().sum()
        stats.loc['missing_%'] = (self.df[self.numeric_cols].isnull().sum() / len(self.df) * 100).round(2)
        stats.loc['zeros'] = (self.df[self.numeric_cols] == 0).sum()
        stats.loc['zeros_%'] = ((self.df[self.numeric_cols] == 0).sum() / len(self.df) * 100).round(2)
        stats.loc['unique'] = self.df[self.numeric_cols].nunique()

        return stats

    def detect_outliers(self, method='iqr', threshold=1.5) -> Dict[str, int]:
        """
        Detect outliers using IQR or Z-score method

        Args:
            method: 'iqr' or 'zscore'
            threshold: IQR multiplier (default 1.5) or Z-score threshold (default 3)

        Returns:
            Dictionary with outlier counts per column
        """
        outliers = {}

        for col in self.numeric_cols:
            if self.df[col].isnull().all():
                continue

            if method == 'iqr':
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                outlier_mask = (self.df[col] < lower_bound) | (self.df[col] > upper_bound)

            elif method == 'zscore':
                z_scores = np.abs((self.df[col] - self.df[col].mean()) / self.df[col].std())
                outlier_mask = z_scores > threshold

            outliers[col] = outlier_mask.sum()

        return outliers

    def calculate_correlation_matrix(self) -> pd.DataFrame:
        """Calculate correlation matrix for numeric columns"""
        return self.df[self.numeric_cols].corr()

    def identify_highly_correlated_pairs(self, threshold=0.8) -> List[Tuple[str, str, float]]:
        """
        Identify highly correlated variable pairs

        Args:
            threshold: Correlation threshold (default 0.8)

        Returns:
            List of tuples (var1, var2, correlation)
        """
        corr_matrix = self.calculate_correlation_matrix()
        high_corr = []

        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) >= threshold:
                    high_corr.append((
                        corr_matrix.columns[i],
                        corr_matrix.columns[j],
                        round(corr_matrix.iloc[i, j], 3)
                    ))

        return sorted(high_corr, key=lambda x: abs(x[2]), reverse=True)

    def analyze_target_correlations(self, target_vars: List[str]) -> pd.DataFrame:
        """
        Analyze correlations with target variables

        Args:
            target_vars: List of target variable names

        Returns:
            DataFrame with correlations sorted by absolute value
        """
        available_targets = [t for t in target_vars if t in self.df.columns]

        if not available_targets:
            raise ValueError("No target variables found in dataset")

        corr_matrix = self.calculate_correlation_matrix()
        target_corrs = corr_matrix[available_targets].copy()

        # Sort by absolute correlation with first target
        target_corrs['abs_corr'] = target_corrs[available_targets[0]].abs()
        target_corrs = target_corrs.sort_values('abs_corr', ascending=False)
        target_corrs = target_corrs.drop('abs_corr', axis=1)

        return target_corrs

    def check_data_quality_issues(self) -> Dict:
        """Comprehensive data quality check"""
        issues = {}

        # Missing values
        missing = self.df.isnull().sum()
        if missing.sum() > 0:
            issues['missing_values'] = missing[missing > 0].to_dict()

        # Constant columns
        constant_cols = [col for col in self.df.columns if self.df[col].nunique() <= 1]
        if constant_cols:
            issues['constant_columns'] = constant_cols

        # Columns with >50% zeros
        high_zero_cols = []
        for col in self.numeric_cols:
            zero_pct = (self.df[col] == 0).sum() / len(self.df) * 100
            if zero_pct > 50:
                high_zero_cols.append((col, round(zero_pct, 2)))
        if high_zero_cols:
            issues['high_zero_percentage'] = high_zero_cols

        # Check for infinite values
        inf_cols = {}
        for col in self.numeric_cols:
            inf_count = np.isinf(self.df[col]).sum()
            if inf_count > 0:
                inf_cols[col] = inf_count
        if inf_cols:
            issues['infinite_values'] = inf_cols

        # Check for negative values where they shouldn't be
        # (temperatures in Celsius can be negative, but flows shouldn't be)
        flow_cols = [col for col in self.df.columns if 'F' in col or 'flow' in col.lower()]
        negative_flow = {}
        for col in flow_cols:
            if col in self.numeric_cols:
                neg_count = (self.df[col] < 0).sum()
                if neg_count > 0:
                    negative_flow[col] = neg_count
        if negative_flow:
            issues['negative_flow_values'] = negative_flow

        return issues

    def get_variable_ranges(self) -> pd.DataFrame:
        """Get min, max, range for all numeric variables"""
        ranges = pd.DataFrame({
            'min': self.df[self.numeric_cols].min(),
            'max': self.df[self.numeric_cols].max(),
            'range': self.df[self.numeric_cols].max() - self.df[self.numeric_cols].min(),
            'mean': self.df[self.numeric_cols].mean(),
            'std': self.df[self.numeric_cols].std()
        })

        return ranges.round(3)


def print_eda_summary(df: pd.DataFrame):
    """Print comprehensive EDA summary"""
    analyzer = EDAAnalyzer(df)

    print("=" * 80)
    print(" EXPLORATORY DATA ANALYSIS SUMMARY")
    print("=" * 80)

    print("\n1. DATASET OVERVIEW")
    print("-" * 80)
    print(f"   Rows: {len(df):,}")
    print(f"   Columns: {len(df.columns)}")
    print(f"   Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print(f"   Numeric Columns: {len(analyzer.numeric_cols)}")

    print("\n2. MISSING VALUES")
    print("-" * 80)
    missing = df.isnull().sum()
    if missing.sum() > 0:
        for col, count in missing[missing > 0].items():
            pct = count / len(df) * 100
            print(f"   {col}: {count:,} ({pct:.2f}%)")
    else:
        print("   ✓ No missing values detected")

    print("\n3. DATA QUALITY ISSUES")
    print("-" * 80)
    issues = analyzer.check_data_quality_issues()
    if issues:
        for issue_type, issue_data in issues.items():
            print(f"   {issue_type}:")
            if isinstance(issue_data, dict):
                for k, v in issue_data.items():
                    print(f"      - {k}: {v}")
            elif isinstance(issue_data, list):
                for item in issue_data:
                    print(f"      - {item}")
    else:
        print("   ✓ No major data quality issues detected")

    print("\n4. OUTLIERS (IQR method, threshold=1.5)")
    print("-" * 80)
    outliers = analyzer.detect_outliers(method='iqr', threshold=1.5)
    outliers_with_count = {k: v for k, v in outliers.items() if v > 0}
    if outliers_with_count:
        for col, count in sorted(outliers_with_count.items(), key=lambda x: x[1], reverse=True)[:10]:
            pct = count / len(df) * 100
            print(f"   {col}: {count:,} ({pct:.2f}%)")
        if len(outliers_with_count) > 10:
            print(f"   ... and {len(outliers_with_count) - 10} more columns with outliers")
    else:
        print("   No outliers detected")

    print("\n5. KEY VARIABLE STATISTICS")
    print("-" * 80)
    key_vars = ['UCWIT', 'UCWOT', 'UCAIT', 'UCAOT', 'UCWF', 'UCAF', 'UCAIH', 'AMBT']
    available_key_vars = [v for v in key_vars if v in df.columns]

    if available_key_vars:
        stats = df[available_key_vars].describe().T[['mean', 'std', 'min', 'max']]
        print(stats.round(2))

    print("\n6. HIGHLY CORRELATED PAIRS (|r| >= 0.8)")
    print("-" * 80)
    high_corr = analyzer.identify_highly_correlated_pairs(threshold=0.8)
    if high_corr:
        for var1, var2, corr in high_corr[:10]:
            print(f"   {var1} <-> {var2}: {corr:.3f}")
        if len(high_corr) > 10:
            print(f"   ... and {len(high_corr) - 10} more pairs")
    else:
        print("   No highly correlated pairs found")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    # Test EDA utilities
    pass
