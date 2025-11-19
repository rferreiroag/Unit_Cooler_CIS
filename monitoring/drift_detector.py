"""
Drift Detection System for HVAC Digital Twin

Monitors data drift and model performance degradation using statistical tests:
- Kolmogorov-Smirnov test for distribution drift
- Population Stability Index (PSI) for feature drift
- Performance monitoring (MAE, RMSE drift)
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple, Optional
import json
from pathlib import Path
from datetime import datetime
import warnings


class DriftDetector:
    """
    Statistical drift detection for monitoring model performance.

    Implements multiple drift detection methods:
    1. Kolmogorov-Smirnov test - distribution changes
    2. Population Stability Index (PSI) - feature drift
    3. Performance drift - prediction accuracy changes
    """

    def __init__(
        self,
        reference_data: pd.DataFrame,
        feature_names: List[str],
        psi_threshold: float = 0.2,
        ks_threshold: float = 0.05,
        performance_threshold: float = 0.1
    ):
        """
        Initialize drift detector.

        Args:
            reference_data: Reference (training) dataset
            feature_names: List of feature names to monitor
            psi_threshold: PSI threshold (>0.2: significant drift)
            ks_threshold: KS test p-value threshold
            performance_threshold: Performance degradation threshold
        """
        self.reference_data = reference_data
        self.feature_names = feature_names
        self.psi_threshold = psi_threshold
        self.ks_threshold = ks_threshold
        self.performance_threshold = performance_threshold

        # Compute reference statistics
        self._compute_reference_stats()

    def _compute_reference_stats(self):
        """Compute statistics on reference data."""
        self.reference_stats = {}

        for feature in self.feature_names:
            if feature in self.reference_data.columns:
                data = self.reference_data[feature].dropna()

                self.reference_stats[feature] = {
                    'mean': data.mean(),
                    'std': data.std(),
                    'min': data.min(),
                    'max': data.max(),
                    'quantiles': np.percentile(data, [0, 25, 50, 75, 100]),
                    'distribution': data.values
                }

    def calculate_psi(
        self,
        reference: np.ndarray,
        current: np.ndarray,
        bins: int = 10
    ) -> float:
        """
        Calculate Population Stability Index (PSI).

        PSI interpretation:
        - <0.1: No significant change
        - 0.1-0.2: Slight change
        - >0.2: Significant change (action required)

        Args:
            reference: Reference distribution
            current: Current distribution
            bins: Number of bins for histogram

        Returns:
            PSI value
        """
        # Create bins based on reference data
        breakpoints = np.percentile(reference, np.linspace(0, 100, bins + 1))
        breakpoints = np.unique(breakpoints)  # Remove duplicates

        # Calculate distributions
        ref_hist, _ = np.histogram(reference, bins=breakpoints)
        cur_hist, _ = np.histogram(current, bins=breakpoints)

        # Normalize to percentages
        ref_pct = ref_hist / len(reference)
        cur_pct = cur_hist / len(current)

        # Avoid division by zero
        ref_pct = np.where(ref_pct == 0, 0.0001, ref_pct)
        cur_pct = np.where(cur_pct == 0, 0.0001, cur_pct)

        # Calculate PSI
        psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))

        return psi

    def ks_test(
        self,
        reference: np.ndarray,
        current: np.ndarray
    ) -> Tuple[float, float]:
        """
        Perform Kolmogorov-Smirnov test.

        Tests if two distributions are significantly different.

        Args:
            reference: Reference distribution
            current: Current distribution

        Returns:
            Tuple of (statistic, p-value)
        """
        statistic, p_value = stats.ks_2samp(reference, current)
        return statistic, p_value

    def detect_feature_drift(
        self,
        current_data: pd.DataFrame
    ) -> Dict[str, Dict]:
        """
        Detect drift in features using PSI and KS test.

        Args:
            current_data: Current dataset to compare

        Returns:
            Dictionary with drift results per feature
        """
        drift_results = {}

        for feature in self.feature_names:
            if feature not in self.reference_data.columns:
                continue

            if feature not in current_data.columns:
                drift_results[feature] = {
                    'status': 'ERROR',
                    'message': 'Feature not found in current data'
                }
                continue

            ref_data = self.reference_data[feature].dropna().values
            cur_data = current_data[feature].dropna().values

            if len(cur_data) == 0:
                drift_results[feature] = {
                    'status': 'ERROR',
                    'message': 'No valid data in current dataset'
                }
                continue

            # Calculate PSI
            psi = self.calculate_psi(ref_data, cur_data)

            # Perform KS test
            ks_stat, ks_pvalue = self.ks_test(ref_data, cur_data)

            # Determine drift status
            if psi > self.psi_threshold and ks_pvalue < self.ks_threshold:
                status = 'CRITICAL'
            elif psi > self.psi_threshold or ks_pvalue < self.ks_threshold:
                status = 'WARNING'
            else:
                status = 'STABLE'

            # Current statistics
            cur_mean = cur_data.mean()
            cur_std = cur_data.std()

            # Changes from reference
            mean_change = (cur_mean - self.reference_stats[feature]['mean']) / \
                         self.reference_stats[feature]['mean'] * 100

            drift_results[feature] = {
                'status': status,
                'psi': round(psi, 4),
                'ks_statistic': round(ks_stat, 4),
                'ks_pvalue': round(ks_pvalue, 4),
                'reference_mean': round(self.reference_stats[feature]['mean'], 4),
                'current_mean': round(cur_mean, 4),
                'mean_change_pct': round(mean_change, 2),
                'reference_std': round(self.reference_stats[feature]['std'], 4),
                'current_std': round(cur_std, 4)
            }

        return drift_results

    def detect_prediction_drift(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        reference_metrics: Dict[str, float]
    ) -> Dict[str, any]:
        """
        Detect drift in prediction performance.

        Args:
            y_true: True values
            y_pred: Predicted values
            reference_metrics: Reference performance metrics

        Returns:
            Dictionary with performance drift results
        """
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

        # Calculate current metrics
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)

        # Calculate drift from reference
        mae_drift = (mae - reference_metrics.get('mae', mae)) / \
                   reference_metrics.get('mae', 1) * 100
        rmse_drift = (rmse - reference_metrics.get('rmse', rmse)) / \
                    reference_metrics.get('rmse', 1) * 100
        r2_drift = reference_metrics.get('r2', r2) - r2

        # Determine status
        if abs(mae_drift) > self.performance_threshold * 100 or \
           abs(rmse_drift) > self.performance_threshold * 100 or \
           abs(r2_drift) > 0.05:
            status = 'WARNING'
        else:
            status = 'STABLE'

        return {
            'status': status,
            'mae': round(mae, 4),
            'rmse': round(rmse, 4),
            'r2': round(r2, 4),
            'mae_drift_pct': round(mae_drift, 2),
            'rmse_drift_pct': round(rmse_drift, 2),
            'r2_drift': round(r2_drift, 4),
            'reference_mae': reference_metrics.get('mae', None),
            'reference_rmse': reference_metrics.get('rmse', None),
            'reference_r2': reference_metrics.get('r2', None)
        }

    def generate_drift_report(
        self,
        current_data: pd.DataFrame,
        y_true: Optional[np.ndarray] = None,
        y_pred: Optional[np.ndarray] = None,
        reference_metrics: Optional[Dict] = None
    ) -> Dict:
        """
        Generate comprehensive drift report.

        Args:
            current_data: Current dataset
            y_true: True values (optional)
            y_pred: Predicted values (optional)
            reference_metrics: Reference performance metrics (optional)

        Returns:
            Complete drift report
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {},
            'feature_drift': {},
            'prediction_drift': None
        }

        # Feature drift detection
        feature_drift = self.detect_feature_drift(current_data)
        report['feature_drift'] = feature_drift

        # Count drift statuses
        critical_count = sum(1 for f in feature_drift.values() if f.get('status') == 'CRITICAL')
        warning_count = sum(1 for f in feature_drift.values() if f.get('status') == 'WARNING')
        stable_count = sum(1 for f in feature_drift.values() if f.get('status') == 'STABLE')

        # Prediction drift (if data provided)
        if y_true is not None and y_pred is not None and reference_metrics is not None:
            pred_drift = self.detect_prediction_drift(y_true, y_pred, reference_metrics)
            report['prediction_drift'] = pred_drift

        # Overall summary
        if critical_count > 0:
            overall_status = 'CRITICAL'
        elif warning_count > 0:
            overall_status = 'WARNING'
        else:
            overall_status = 'STABLE'

        report['summary'] = {
            'overall_status': overall_status,
            'total_features': len(feature_drift),
            'critical_features': critical_count,
            'warning_features': warning_count,
            'stable_features': stable_count,
            'drift_percentage': round((critical_count + warning_count) / len(feature_drift) * 100, 2)
        }

        return report

    def save_report(self, report: Dict, output_path: str):
        """Save drift report to JSON file."""
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)


def load_reference_data(data_path: str) -> pd.DataFrame:
    """Load reference data from CSV."""
    return pd.read_csv(data_path)


def example_usage():
    """Example usage of drift detector."""
    print("="*70)
    print("DRIFT DETECTION SYSTEM - EXAMPLE")
    print("="*70)

    # Load reference data (training data)
    project_root = Path(__file__).parent.parent
    data_path = project_root / 'data' / 'processed' / 'X_train.csv'

    if not data_path.exists():
        print(f"Error: Reference data not found at {data_path}")
        return

    print(f"\nLoading reference data from: {data_path}")
    reference_data = pd.read_csv(data_path)

    # Select key features to monitor
    key_features = ['UCWIT', 'UCAIT', 'UCWF', 'UCAIH', 'AMBT', 'UCTSP']
    available_features = [f for f in key_features if f in reference_data.columns]

    print(f"Monitoring {len(available_features)} features: {available_features}")

    # Initialize drift detector
    detector = DriftDetector(
        reference_data=reference_data,
        feature_names=available_features,
        psi_threshold=0.2,
        ks_threshold=0.05
    )

    # Simulate current data (using validation set)
    val_data_path = project_root / 'data' / 'processed' / 'X_val.csv'
    if val_data_path.exists():
        print(f"\nSimulating drift detection with validation data...")
        current_data = pd.read_csv(val_data_path)

        # Generate drift report
        report = detector.generate_drift_report(current_data)

        # Print summary
        print(f"\n{'='*70}")
        print(f"DRIFT REPORT SUMMARY")
        print(f"{'='*70}")
        print(f"Overall Status: {report['summary']['overall_status']}")
        print(f"Total Features: {report['summary']['total_features']}")
        print(f"Critical: {report['summary']['critical_features']}")
        print(f"Warning: {report['summary']['warning_features']}")
        print(f"Stable: {report['summary']['stable_features']}")
        print(f"Drift Percentage: {report['summary']['drift_percentage']}%")

        # Print feature details
        print(f"\n{'='*70}")
        print(f"FEATURE DRIFT DETAILS")
        print(f"{'='*70}")
        print(f"{'Feature':<15} {'Status':<12} {'PSI':<10} {'Mean Change':<15}")
        print(f"{'-'*70}")

        for feature, result in report['feature_drift'].items():
            if result.get('status') != 'ERROR':
                print(f"{feature:<15} {result['status']:<12} {result['psi']:<10.4f} "
                      f"{result['mean_change_pct']:<15.2f}%")

        # Save report
        output_path = project_root / 'monitoring' / 'drift_report.json'
        detector.save_report(report, str(output_path))
        print(f"\n✓ Report saved to: {output_path}")

    else:
        print(f"Warning: Validation data not found at {val_data_path}")


if __name__ == '__main__':
    example_usage()
