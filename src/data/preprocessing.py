"""
Data preprocessing pipeline for Unit Cooler Digital Twin
Handles cleaning, imputation, and validation of experimental data
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler, RobustScaler
import warnings
warnings.filterwarnings('ignore')


class DataPreprocessor:
    """Comprehensive preprocessing pipeline for Unit Cooler data"""

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize preprocessor

        Args:
            config: Configuration dictionary with preprocessing parameters
        """
        self.config = config or self._default_config()
        self.statistics = {}
        self.is_fitted = False

    def _default_config(self) -> Dict:
        """Default preprocessing configuration"""
        return {
            # Temperature limits (°C)
            'temp_min': -10.0,
            'temp_max': 150.0,

            # Flow limits
            'flow_min': 0.0,
            'flow_max': 20000.0,

            # Humidity limits (%)
            'humidity_min': 0.0,
            'humidity_max': 100.0,

            # Pressure limits
            'pressure_min': 0.0,
            'pressure_max': 500.0,

            # Sensor saturation values to handle
            'saturation_values': [65535, 65534, 0, -999, -9999],

            # Missing value threshold (drop columns with > threshold)
            'missing_threshold': 0.70,

            # Imputation strategy
            'imputation_strategy': 'forward_fill',  # or 'mean', 'median', 'interpolate'

            # Outlier handling (IQR multiplier)
            'outlier_iqr_multiplier': 3.0,  # More lenient than 1.5

            # Scaling method
            'scaling_method': 'standard'  # or 'robust', 'minmax'
        }

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit preprocessor and transform data

        Args:
            df: Input DataFrame

        Returns:
            Preprocessed DataFrame
        """
        self.is_fitted = False
        df_clean = self._preprocess_pipeline(df, fit=True)
        self.is_fitted = True
        return df_clean

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data using fitted preprocessor

        Args:
            df: Input DataFrame

        Returns:
            Preprocessed DataFrame
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor not fitted. Call fit_transform first.")

        return self._preprocess_pipeline(df, fit=False)

    def _preprocess_pipeline(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Main preprocessing pipeline

        Args:
            df: Input DataFrame
            fit: Whether to fit statistics or use existing

        Returns:
            Preprocessed DataFrame
        """
        print("\n" + "="*80)
        print(" DATA PREPROCESSING PIPELINE")
        print("="*80)
        print(f"\nInput shape: {df.shape}")

        df_clean = df.copy()

        # Step 1: Handle sensor saturation
        print("\n[1/8] Handling sensor saturation...")
        df_clean = self._handle_sensor_saturation(df_clean)

        # Step 2: Fix negative flows
        print("[2/8] Fixing negative flow values...")
        df_clean = self._fix_negative_flows(df_clean)

        # Step 3: Clip extreme values
        print("[3/8] Clipping extreme values to physical limits...")
        df_clean = self._clip_extreme_values(df_clean)

        # Step 4: Drop high-missing columns
        print("[4/8] Dropping columns with excessive missing data...")
        df_clean = self._drop_high_missing_columns(df_clean)

        # Step 5: Impute remaining missing values
        print("[5/8] Imputing remaining missing values...")
        df_clean = self._impute_missing_values(df_clean, fit=fit)

        # Step 6: Handle outliers
        print("[6/8] Handling outliers...")
        df_clean = self._handle_outliers(df_clean, fit=fit)

        # Step 7: Validate physics constraints
        print("[7/8] Validating physics constraints...")
        df_clean = self._validate_physics_constraints(df_clean)

        # Step 8: Final validation
        print("[8/8] Final validation...")
        df_clean = self._final_validation(df_clean)

        print(f"\nOutput shape: {df_clean.shape}")
        print(f"Samples removed: {len(df) - len(df_clean)} ({(len(df) - len(df_clean))/len(df)*100:.2f}%)")
        print("="*80)

        return df_clean

    def _handle_sensor_saturation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Replace sensor saturation values with NaN"""
        df_clean = df.copy()

        for value in self.config['saturation_values']:
            mask = df_clean == value
            count = mask.sum().sum()
            if count > 0:
                print(f"  Replacing {count} occurrences of {value} with NaN")
                df_clean = df_clean.replace(value, np.nan)

        # Special handling for UCAF (air flow) = 65535
        if 'UCAF' in df_clean.columns:
            ucaf_saturated = (df_clean['UCAF'] >= 65534).sum()
            if ucaf_saturated > 0:
                print(f"  UCAF sensor saturation: {ucaf_saturated} samples (setting to NaN)")
                df_clean.loc[df_clean['UCAF'] >= 65534, 'UCAF'] = np.nan

        return df_clean

    def _fix_negative_flows(self, df: pd.DataFrame) -> pd.DataFrame:
        """Set negative flow values to zero or NaN"""
        df_clean = df.copy()

        # Flow-related columns (ending with F or containing 'flow')
        flow_cols = [col for col in df_clean.columns
                     if col.endswith('F') or 'flow' in col.lower() or 'F' in col]

        for col in flow_cols:
            if col in df_clean.columns:
                negative_count = (df_clean[col] < 0).sum()
                if negative_count > 0:
                    print(f"  {col}: {negative_count} negative values → set to 0")
                    df_clean.loc[df_clean[col] < 0, col] = 0.0

        return df_clean

    def _clip_extreme_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clip values to physical limits"""
        df_clean = df.copy()
        clipped_counts = {}

        # Temperature columns (ending with T or containing 'temp')
        temp_cols = [col for col in df_clean.columns
                     if col.endswith('T') or 'temp' in col.lower() or 'AMBT' in col]

        for col in temp_cols:
            if col in df_clean.columns:
                original = df_clean[col].copy()
                df_clean[col] = df_clean[col].clip(
                    lower=self.config['temp_min'],
                    upper=self.config['temp_max']
                )
                clipped = ((original != df_clean[col]) & original.notna()).sum()
                if clipped > 0:
                    clipped_counts[col] = clipped

        # Flow columns
        flow_cols = [col for col in df_clean.columns
                     if col.endswith('F') or 'flow' in col.lower()]

        for col in flow_cols:
            if col in df_clean.columns:
                original = df_clean[col].copy()
                df_clean[col] = df_clean[col].clip(
                    lower=self.config['flow_min'],
                    upper=self.config['flow_max']
                )
                clipped = ((original != df_clean[col]) & original.notna()).sum()
                if clipped > 0:
                    clipped_counts[col] = clipped

        # Humidity columns
        if 'UCAIH' in df_clean.columns:
            original = df_clean['UCAIH'].copy()
            df_clean['UCAIH'] = df_clean['UCAIH'].clip(
                lower=self.config['humidity_min'],
                upper=self.config['humidity_max']
            )
            clipped = ((original != df_clean['UCAIH']) & original.notna()).sum()
            if clipped > 0:
                clipped_counts['UCAIH'] = clipped

        if clipped_counts:
            print(f"  Clipped extreme values:")
            for col, count in sorted(clipped_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
                print(f"    {col}: {count} values")

        return df_clean

    def _drop_high_missing_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop columns with excessive missing data"""
        df_clean = df.copy()

        missing_pct = df_clean.isnull().sum() / len(df_clean)
        high_missing_cols = missing_pct[missing_pct > self.config['missing_threshold']].index.tolist()

        if high_missing_cols:
            print(f"  Dropping {len(high_missing_cols)} columns with >{self.config['missing_threshold']*100}% missing:")
            for col in high_missing_cols:
                pct = missing_pct[col] * 100
                print(f"    {col}: {pct:.1f}% missing")

            df_clean = df_clean.drop(columns=high_missing_cols)
        else:
            print(f"  No columns with >{self.config['missing_threshold']*100}% missing data")

        return df_clean

    def _impute_missing_values(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Impute remaining missing values"""
        df_clean = df.copy()

        strategy = self.config['imputation_strategy']

        missing_before = df_clean.isnull().sum().sum()

        if strategy == 'forward_fill':
            # Forward fill for time series continuity
            df_clean = df_clean.fillna(method='ffill', limit=5)
            # Backward fill for remaining
            df_clean = df_clean.fillna(method='bfill', limit=5)
            # Mean for any remaining
            if fit:
                self.statistics['column_means'] = df_clean.mean()
            df_clean = df_clean.fillna(self.statistics.get('column_means', df_clean.mean()))

        elif strategy == 'mean':
            if fit:
                self.statistics['column_means'] = df_clean.mean()
            df_clean = df_clean.fillna(self.statistics.get('column_means', df_clean.mean()))

        elif strategy == 'median':
            if fit:
                self.statistics['column_medians'] = df_clean.median()
            df_clean = df_clean.fillna(self.statistics.get('column_medians', df_clean.median()))

        elif strategy == 'interpolate':
            df_clean = df_clean.interpolate(method='linear', limit=10, limit_direction='both')
            # Mean for remaining
            if fit:
                self.statistics['column_means'] = df_clean.mean()
            df_clean = df_clean.fillna(self.statistics.get('column_means', df_clean.mean()))

        missing_after = df_clean.isnull().sum().sum()
        print(f"  Missing values: {missing_before:,} → {missing_after:,} (strategy: {strategy})")

        return df_clean

    def _handle_outliers(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Handle outliers using IQR method (clip instead of remove)"""
        df_clean = df.copy()

        if fit:
            self.statistics['outlier_bounds'] = {}

        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns

        total_clipped = 0

        for col in numeric_cols:
            if fit:
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - self.config['outlier_iqr_multiplier'] * IQR
                upper_bound = Q3 + self.config['outlier_iqr_multiplier'] * IQR

                self.statistics['outlier_bounds'][col] = {
                    'lower': lower_bound,
                    'upper': upper_bound
                }

            bounds = self.statistics['outlier_bounds'][col]

            # Clip outliers instead of removing
            original = df_clean[col].copy()
            df_clean[col] = df_clean[col].clip(lower=bounds['lower'], upper=bounds['upper'])

            clipped = ((original != df_clean[col]) & original.notna()).sum()
            total_clipped += clipped

        print(f"  Clipped {total_clipped:,} outlier values (IQR multiplier: {self.config['outlier_iqr_multiplier']})")

        return df_clean

    def _validate_physics_constraints(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and flag physics constraint violations"""
        df_clean = df.copy()

        violations = {}

        # Check if required columns exist
        required_temps = ['UCWIT', 'UCWOT', 'UCAIT', 'UCAOT']
        available_temps = [col for col in required_temps if col in df_clean.columns]

        if len(available_temps) >= 2:
            # Check water temperature: inlet should be <= outlet for cooling
            # Actually, for cooling, inlet temp > outlet temp (heat is removed)
            # UCWIT > UCWOT for chilled water system
            if 'UCWIT' in df_clean.columns and 'UCWOT' in df_clean.columns:
                # In cooling mode, water gets warmer: UCWOT might be > UCWIT if heating
                # Let's just check for reasonable deltas
                delta_water = df_clean['UCWIT'] - df_clean['UCWOT']
                unrealistic_delta = (delta_water.abs() > 50).sum()
                if unrealistic_delta > 0:
                    violations['water_delta_T'] = unrealistic_delta

            # Check air temperature
            if 'UCAIT' in df_clean.columns and 'UCAOT' in df_clean.columns:
                delta_air = df_clean['UCAOT'] - df_clean['UCAIT']
                unrealistic_delta = (delta_air.abs() > 50).sum()
                if unrealistic_delta > 0:
                    violations['air_delta_T'] = unrealistic_delta

        if violations:
            print(f"  Physics constraint warnings:")
            for constraint, count in violations.items():
                print(f"    {constraint}: {count} samples with extreme values")
        else:
            print(f"  Physics constraints: OK")

        return df_clean

    def _final_validation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Final validation and cleanup"""
        df_clean = df.copy()

        # Remove any remaining NaN rows
        rows_before = len(df_clean)
        df_clean = df_clean.dropna()
        rows_after = len(df_clean)

        if rows_before != rows_after:
            print(f"  Removed {rows_before - rows_after} rows with remaining NaN values")

        # Reset index
        df_clean = df_clean.reset_index(drop=True)

        # Final check
        assert df_clean.isnull().sum().sum() == 0, "NaN values still present!"
        assert len(df_clean) > 0, "No data remaining after preprocessing!"

        print(f"  ✓ Final dataset: {len(df_clean):,} samples, {len(df_clean.columns)} features")

        return df_clean

    def get_preprocessing_report(self) -> Dict:
        """Get detailed preprocessing report"""
        if not self.is_fitted:
            raise ValueError("Preprocessor not fitted yet")

        report = {
            'config': self.config,
            'statistics': self.statistics,
            'is_fitted': self.is_fitted
        }

        return report


def preprocess_unit_cooler_data(df: pd.DataFrame, config: Optional[Dict] = None) -> Tuple[pd.DataFrame, DataPreprocessor]:
    """
    Convenience function to preprocess Unit Cooler data

    Args:
        df: Input DataFrame
        config: Optional preprocessing configuration

    Returns:
        Tuple of (preprocessed DataFrame, fitted preprocessor)
    """
    preprocessor = DataPreprocessor(config=config)
    df_clean = preprocessor.fit_transform(df)

    return df_clean, preprocessor


if __name__ == "__main__":
    # Test preprocessing pipeline
    import sys
    sys.path.append('../..')
    from src.data.data_loader import load_and_preprocess

    # Load data
    df, metadata = load_and_preprocess('../../data/raw/datos_combinados_entrenamiento_20251118_105234.csv')

    # Preprocess
    df_clean, preprocessor = preprocess_unit_cooler_data(df)

    print("\n" + "="*80)
    print(" PREPROCESSING COMPLETE")
    print("="*80)
    print(f"Original: {df.shape}")
    print(f"Cleaned: {df_clean.shape}")
    print(f"Data retention: {len(df_clean)/len(df)*100:.2f}%")
