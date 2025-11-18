"""
Temporal data splitting and normalization for Unit Cooler Digital Twin
Ensures no data leakage for time series modeling
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, List, Optional
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
import warnings
warnings.filterwarnings('ignore')


class TemporalDataSplitter:
    """Handle temporal train/val/test splits without data leakage"""

    def __init__(self, train_ratio: float = 0.70, val_ratio: float = 0.15, test_ratio: float = 0.15):
        """
        Initialize temporal splitter

        Args:
            train_ratio: Proportion for training set
            val_ratio: Proportion for validation set
            test_ratio: Proportion for test set
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            "Ratios must sum to 1.0"

        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

    def split(self, df: pd.DataFrame, target_cols: List[str],
              shuffle: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame,
                                               pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data temporally into train/val/test

        Args:
            df: Input DataFrame
            target_cols: List of target column names
            shuffle: Whether to shuffle (NOT recommended for time series)

        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        print("\n" + "="*80)
        print(" TEMPORAL DATA SPLITTING")
        print("="*80)

        if shuffle:
            print("\n⚠️  WARNING: Shuffling time series data may cause data leakage!")

        n_samples = len(df)

        # Calculate split indices
        train_end = int(n_samples * self.train_ratio)
        val_end = int(n_samples * (self.train_ratio + self.val_ratio))

        print(f"\nDataset size: {n_samples:,} samples")
        print(f"Split ratios: Train={self.train_ratio:.0%}, Val={self.val_ratio:.0%}, Test={self.test_ratio:.0%}")

        if shuffle:
            df_shuffled = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
        else:
            df_shuffled = df.copy()

        # Split temporally
        df_train = df_shuffled.iloc[:train_end].copy()
        df_val = df_shuffled.iloc[train_end:val_end].copy()
        df_test = df_shuffled.iloc[val_end:].copy()

        print(f"\nSplit sizes:")
        print(f"  Train: {len(df_train):,} samples ({len(df_train)/n_samples*100:.1f}%)")
        print(f"  Val:   {len(df_val):,} samples ({len(df_val)/n_samples*100:.1f}%)")
        print(f"  Test:  {len(df_test):,} samples ({len(df_test)/n_samples*100:.1f}%)")

        # Separate features and targets
        available_targets = [col for col in target_cols if col in df.columns]
        feature_cols = [col for col in df.columns if col not in available_targets]

        print(f"\nFeatures: {len(feature_cols)}")
        print(f"Targets: {len(available_targets)} ({available_targets})")

        X_train = df_train[feature_cols]
        X_val = df_val[feature_cols]
        X_test = df_test[feature_cols]

        y_train = df_train[available_targets]
        y_val = df_val[available_targets]
        y_test = df_test[available_targets]

        print("\n✓ Temporal split complete (no data leakage)")
        print("="*80)

        return X_train, X_val, X_test, y_train, y_val, y_test

    def create_sequences(self, X: pd.DataFrame, y: pd.DataFrame,
                        sequence_length: int = 10,
                        prediction_horizon: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM/GRU models

        Args:
            X: Input features
            y: Target variables
            sequence_length: Number of past timesteps to use
            prediction_horizon: Number of future steps to predict

        Returns:
            Tuple of (X_sequences, y_sequences)
        """
        X_seq = []
        y_seq = []

        X_array = X.values
        y_array = y.values

        for i in range(len(X) - sequence_length - prediction_horizon + 1):
            X_seq.append(X_array[i:i+sequence_length])
            y_seq.append(y_array[i+sequence_length+prediction_horizon-1])

        return np.array(X_seq), np.array(y_seq)


class AdaptiveScaler:
    """Adaptive scaling by operational regime"""

    def __init__(self, scaling_method: str = 'standard',
                 regime_column: Optional[str] = None):
        """
        Initialize adaptive scaler

        Args:
            scaling_method: 'standard', 'robust', or 'minmax'
            regime_column: Column name to identify regimes (optional)
        """
        self.scaling_method = scaling_method
        self.regime_column = regime_column
        self.scaler_X = None
        self.scaler_y = None
        self.regime_scalers_X = {}
        self.regime_scalers_y = {}
        self.is_fitted = False

    def _create_scaler(self):
        """Create scaler based on method"""
        if self.scaling_method == 'standard':
            return StandardScaler()
        elif self.scaling_method == 'robust':
            return RobustScaler()
        elif self.scaling_method == 'minmax':
            return MinMaxScaler()
        else:
            raise ValueError(f"Unknown scaling method: {self.scaling_method}")

    def fit(self, X_train: pd.DataFrame, y_train: pd.DataFrame):
        """
        Fit scalers on training data

        Args:
            X_train: Training features
            y_train: Training targets
        """
        print("\n" + "="*80)
        print(" ADAPTIVE SCALING")
        print("="*80)
        print(f"\nScaling method: {self.scaling_method}")

        if self.regime_column is not None and self.regime_column in X_train.columns:
            print(f"Regime-based scaling: {self.regime_column}")

            # Identify unique regimes
            regimes = X_train[self.regime_column].unique()
            print(f"Number of regimes: {len(regimes)}")

            for regime in regimes:
                # Filter by regime
                mask = X_train[self.regime_column] == regime
                X_regime = X_train[mask]
                y_regime = y_train[mask]

                if len(X_regime) > 10:  # Minimum samples for regime
                    # Create and fit scalers for this regime
                    scaler_X = self._create_scaler()
                    scaler_y = self._create_scaler()

                    # Drop regime column before scaling
                    X_regime_no_label = X_regime.drop(columns=[self.regime_column])

                    scaler_X.fit(X_regime_no_label)
                    scaler_y.fit(y_regime)

                    self.regime_scalers_X[regime] = scaler_X
                    self.regime_scalers_y[regime] = scaler_y

                    print(f"  Regime {regime}: {len(X_regime)} samples")

        else:
            print("Global scaling (no regime separation)")

            # Global scalers
            self.scaler_X = self._create_scaler()
            self.scaler_y = self._create_scaler()

            # Drop regime column if present
            X_to_scale = X_train.drop(columns=[self.regime_column], errors='ignore')

            self.scaler_X.fit(X_to_scale)
            self.scaler_y.fit(y_train)

        self.is_fitted = True
        print("\n✓ Scalers fitted successfully")
        print("="*80)

    def transform(self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None,
                  regime: Optional[str] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Transform data using fitted scalers

        Args:
            X: Input features
            y: Target variables (optional)
            regime: Specific regime to use (optional)

        Returns:
            Tuple of (X_scaled, y_scaled)
        """
        if not self.is_fitted:
            raise ValueError("Scaler not fitted. Call fit() first.")

        # Drop regime column if present
        X_to_scale = X.drop(columns=[self.regime_column], errors='ignore')

        if regime is not None and regime in self.regime_scalers_X:
            # Use regime-specific scaler
            X_scaled = self.regime_scalers_X[regime].transform(X_to_scale)
            y_scaled = self.regime_scalers_y[regime].transform(y) if y is not None else None

        elif self.scaler_X is not None:
            # Use global scaler
            X_scaled = self.scaler_X.transform(X_to_scale)
            y_scaled = self.scaler_y.transform(y) if y is not None else None

        else:
            raise ValueError("No scaler available for transformation")

        return X_scaled, y_scaled

    def inverse_transform_y(self, y_scaled: np.ndarray,
                           regime: Optional[str] = None) -> np.ndarray:
        """
        Inverse transform target variables

        Args:
            y_scaled: Scaled target variables
            regime: Specific regime (optional)

        Returns:
            Original scale targets
        """
        if not self.is_fitted:
            raise ValueError("Scaler not fitted. Call fit() first.")

        if regime is not None and regime in self.regime_scalers_y:
            return self.regime_scalers_y[regime].inverse_transform(y_scaled)
        elif self.scaler_y is not None:
            return self.scaler_y.inverse_transform(y_scaled)
        else:
            raise ValueError("No scaler available for inverse transformation")


def prepare_temporal_data(df: pd.DataFrame, target_cols: List[str],
                         train_ratio: float = 0.70,
                         val_ratio: float = 0.15,
                         test_ratio: float = 0.15,
                         scaling_method: str = 'standard',
                         shuffle: bool = False) -> Dict:
    """
    Complete data preparation pipeline with temporal split and scaling

    Args:
        df: Input DataFrame (preprocessed with features)
        target_cols: List of target column names
        train_ratio: Training set proportion
        val_ratio: Validation set proportion
        test_ratio: Test set proportion
        scaling_method: Scaling method ('standard', 'robust', 'minmax')
        shuffle: Whether to shuffle (NOT recommended for time series)

    Returns:
        Dictionary with all data splits and scalers
    """
    # Split data
    splitter = TemporalDataSplitter(train_ratio, val_ratio, test_ratio)
    X_train, X_val, X_test, y_train, y_val, y_test = splitter.split(df, target_cols, shuffle=shuffle)

    # Scale data
    scaler = AdaptiveScaler(scaling_method=scaling_method)
    scaler.fit(X_train, y_train)

    X_train_scaled, y_train_scaled = scaler.transform(X_train, y_train)
    X_val_scaled, y_val_scaled = scaler.transform(X_val, y_val)
    X_test_scaled, y_test_scaled = scaler.transform(X_test, y_test)

    return {
        'X_train': X_train,
        'X_val': X_val,
        'X_test': X_test,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test,
        'X_train_scaled': X_train_scaled,
        'X_val_scaled': X_val_scaled,
        'X_test_scaled': X_test_scaled,
        'y_train_scaled': y_train_scaled,
        'y_val_scaled': y_val_scaled,
        'y_test_scaled': y_test_scaled,
        'scaler': scaler,
        'splitter': splitter,
        'feature_names': X_train.columns.tolist(),
        'target_names': y_train.columns.tolist()
    }


def save_processed_data(data_dict: Dict, output_dir: str = 'data/processed'):
    """
    Save processed data to disk

    Args:
        data_dict: Dictionary from prepare_temporal_data
        output_dir: Output directory
    """
    import os
    import joblib

    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "="*80)
    print(" SAVING PROCESSED DATA")
    print("="*80)

    # Save DataFrames
    data_dict['X_train'].to_csv(f"{output_dir}/X_train.csv", index=False)
    data_dict['X_val'].to_csv(f"{output_dir}/X_val.csv", index=False)
    data_dict['X_test'].to_csv(f"{output_dir}/X_test.csv", index=False)

    data_dict['y_train'].to_csv(f"{output_dir}/y_train.csv", index=False)
    data_dict['y_val'].to_csv(f"{output_dir}/y_val.csv", index=False)
    data_dict['y_test'].to_csv(f"{output_dir}/y_test.csv", index=False)

    print(f"  ✓ Saved CSV files to {output_dir}/")

    # Save scaled arrays
    np.save(f"{output_dir}/X_train_scaled.npy", data_dict['X_train_scaled'])
    np.save(f"{output_dir}/X_val_scaled.npy", data_dict['X_val_scaled'])
    np.save(f"{output_dir}/X_test_scaled.npy", data_dict['X_test_scaled'])

    np.save(f"{output_dir}/y_train_scaled.npy", data_dict['y_train_scaled'])
    np.save(f"{output_dir}/y_val_scaled.npy", data_dict['y_val_scaled'])
    np.save(f"{output_dir}/y_test_scaled.npy", data_dict['y_test_scaled'])

    print(f"  ✓ Saved scaled arrays to {output_dir}/")

    # Save scaler
    joblib.dump(data_dict['scaler'], f"{output_dir}/scaler.pkl")
    print(f"  ✓ Saved scaler to {output_dir}/scaler.pkl")

    # Save metadata
    metadata = {
        'feature_names': data_dict['feature_names'],
        'target_names': data_dict['target_names'],
        'train_size': len(data_dict['X_train']),
        'val_size': len(data_dict['X_val']),
        'test_size': len(data_dict['X_test']),
        'n_features': len(data_dict['feature_names']),
        'n_targets': len(data_dict['target_names'])
    }

    import json
    with open(f"{output_dir}/metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"  ✓ Saved metadata to {output_dir}/metadata.json")
    print("="*80)


if __name__ == "__main__":
    # Test temporal splitting and scaling
    import sys
    sys.path.append('../..')
    from src.data.data_loader import load_and_preprocess
    from src.data.preprocessing import preprocess_unit_cooler_data
    from src.data.feature_engineering import engineer_features

    # Load, preprocess, and engineer features
    df, _ = load_and_preprocess('../../data/raw/datos_combinados_entrenamiento_20251118_105234.csv')
    df_clean, _ = preprocess_unit_cooler_data(df)
    df_features, _ = engineer_features(df_clean)

    # Prepare temporal data
    target_cols = ['UCAOT', 'UCWOT', 'UCAF']
    data_dict = prepare_temporal_data(df_features, target_cols,
                                      train_ratio=0.70, val_ratio=0.15, test_ratio=0.15,
                                      scaling_method='standard')

    # Save processed data
    save_processed_data(data_dict)

    print("\n" + "="*80)
    print(" DATA PREPARATION COMPLETE")
    print("="*80)
    print(f"Features: {len(data_dict['feature_names'])}")
    print(f"Targets: {len(data_dict['target_names'])}")
    print(f"Train samples: {len(data_dict['X_train']):,}")
    print(f"Val samples: {len(data_dict['X_val']):,}")
    print(f"Test samples: {len(data_dict['X_test']):,}")
