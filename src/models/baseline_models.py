"""
Baseline models for Unit Cooler Digital Twin
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.multioutput import MultiOutputRegressor
import joblib
import json
from pathlib import Path
from typing import Dict, Tuple, List
import warnings
warnings.filterwarnings('ignore')


class BaselineModel:
    """Wrapper class for baseline models"""

    def __init__(self, model_type: str = 'linear', **kwargs):
        """
        Initialize baseline model

        Args:
            model_type: Type of model ('linear' or 'rf')
            **kwargs: Model-specific parameters
        """
        self.model_type = model_type
        self.model = self._create_model(**kwargs)
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.feature_names = None
        self.target_names = None
        self.is_fitted = False

    def _create_model(self, **kwargs):
        """Create the model based on type"""
        if self.model_type == 'linear':
            return MultiOutputRegressor(LinearRegression(**kwargs))
        elif self.model_type == 'rf':
            default_params = {
                'n_estimators': 100,
                'max_depth': 15,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'random_state': 42,
                'n_jobs': -1
            }
            default_params.update(kwargs)
            return MultiOutputRegressor(RandomForestRegressor(**default_params))
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def fit(self, X: pd.DataFrame, y: pd.DataFrame, scale: bool = True):
        """
        Train the model

        Args:
            X: Input features
            y: Target variables
            scale: Whether to scale features and targets
        """
        self.feature_names = list(X.columns)
        self.target_names = list(y.columns)

        if scale:
            X_scaled = self.scaler_X.fit_transform(X)
            y_scaled = self.scaler_y.fit_transform(y)
        else:
            X_scaled = X.values
            y_scaled = y.values

        print(f"Training {self.model_type} model...")
        print(f"  Input shape: {X_scaled.shape}")
        print(f"  Output shape: {y_scaled.shape}")

        self.model.fit(X_scaled, y_scaled)
        self.is_fitted = True

        print(f"✓ Model trained successfully")

    def predict(self, X: pd.DataFrame, scale: bool = True) -> np.ndarray:
        """
        Make predictions

        Args:
            X: Input features
            scale: Whether to scale features

        Returns:
            Predictions array
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")

        if scale:
            X_scaled = self.scaler_X.transform(X)
        else:
            X_scaled = X.values

        y_pred_scaled = self.model.predict(X_scaled)

        if scale:
            y_pred = self.scaler_y.inverse_transform(y_pred_scaled)
        else:
            y_pred = y_pred_scaled

        return y_pred

    def evaluate(self, X: pd.DataFrame, y: pd.DataFrame) -> Dict:
        """
        Evaluate model performance

        Args:
            X: Input features
            y: True target values

        Returns:
            Dictionary with metrics
        """
        y_pred = self.predict(X)

        metrics = {}
        for i, target in enumerate(self.target_names):
            y_true_i = y.iloc[:, i].values
            y_pred_i = y_pred[:, i]

            # Remove NaN values
            mask = ~np.isnan(y_true_i) & ~np.isnan(y_pred_i)
            y_true_clean = y_true_i[mask]
            y_pred_clean = y_pred_i[mask]

            if len(y_true_clean) > 0:
                mae = mean_absolute_error(y_true_clean, y_pred_clean)
                rmse = np.sqrt(mean_squared_error(y_true_clean, y_pred_clean))
                r2 = r2_score(y_true_clean, y_pred_clean)

                # MAPE (avoid division by zero)
                mape_mask = y_true_clean != 0
                if mape_mask.sum() > 0:
                    mape = np.mean(np.abs((y_true_clean[mape_mask] - y_pred_clean[mape_mask]) /
                                         y_true_clean[mape_mask])) * 100
                else:
                    mape = np.nan

                max_error = np.max(np.abs(y_true_clean - y_pred_clean))

                metrics[target] = {
                    'MAE': round(mae, 4),
                    'RMSE': round(rmse, 4),
                    'MAPE': round(mape, 4) if not np.isnan(mape) else None,
                    'R2': round(r2, 4),
                    'Max_Error': round(max_error, 4)
                }
            else:
                metrics[target] = {
                    'MAE': None,
                    'RMSE': None,
                    'MAPE': None,
                    'R2': None,
                    'Max_Error': None
                }

        return metrics

    def save(self, filepath: str):
        """Save model to disk"""
        joblib.dump({
            'model': self.model,
            'scaler_X': self.scaler_X,
            'scaler_y': self.scaler_y,
            'feature_names': self.feature_names,
            'target_names': self.target_names,
            'model_type': self.model_type
        }, filepath)
        print(f"✓ Model saved: {filepath}")

    def load(self, filepath: str):
        """Load model from disk"""
        data = joblib.load(filepath)
        self.model = data['model']
        self.scaler_X = data['scaler_X']
        self.scaler_y = data['scaler_y']
        self.feature_names = data['feature_names']
        self.target_names = data['target_names']
        self.model_type = data['model_type']
        self.is_fitted = True
        print(f"✓ Model loaded: {filepath}")


def prepare_data_for_modeling(df: pd.DataFrame, target_vars: List[str],
                              test_size: float = 0.3, random_state: int = 42) -> Tuple:
    """
    Prepare data for modeling with train/val/test splits

    Args:
        df: Input DataFrame
        target_vars: List of target variable names
        test_size: Proportion for test set
        random_state: Random seed

    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    print("\n=== PREPARING DATA FOR MODELING ===")

    # Remove rows with all NaN targets
    available_targets = [t for t in target_vars if t in df.columns]
    print(f"\nTarget variables: {available_targets}")

    # Drop rows where all targets are NaN
    df_clean = df.dropna(subset=available_targets, how='all').copy()
    print(f"After removing rows with all NaN targets: {len(df_clean):,} rows")

    # Identify feature columns (exclude targets)
    feature_cols = [col for col in df_clean.columns if col not in available_targets]

    # Remove features with too many missing values (>50%)
    missing_pct = df_clean[feature_cols].isnull().sum() / len(df_clean) * 100
    valid_features = missing_pct[missing_pct <= 50].index.tolist()

    print(f"\nFeatures after removing high missing (>50%): {len(valid_features)}")

    # Fill remaining missing values with mean
    X = df_clean[valid_features].copy()
    for col in X.columns:
        X[col].fillna(X[col].mean(), inplace=True)

    y = df_clean[available_targets].copy()

    # Fill target NaN with mean (for training purposes)
    for col in y.columns:
        y[col].fillna(y[col].mean(), inplace=True)

    print(f"\nFinal dataset:")
    print(f"  Samples: {len(X):,}")
    print(f"  Features: {X.shape[1]}")
    print(f"  Targets: {y.shape[1]}")

    # Split into train, val, test (60%, 20%, 20%)
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=test_size,
                                                        random_state=random_state)

    val_size_adjusted = 0.25  # 25% of temp = 20% of original
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=val_size_adjusted,
                                                        random_state=random_state)

    print(f"\nData splits:")
    print(f"  Train: {len(X_train):,} ({len(X_train)/len(X)*100:.1f}%)")
    print(f"  Val:   {len(X_val):,} ({len(X_val)/len(X)*100:.1f}%)")
    print(f"  Test:  {len(X_test):,} ({len(X_test)/len(X)*100:.1f}%)")

    return X_train, X_val, X_test, y_train, y_val, y_test


def train_and_evaluate_baseline_models(X_train: pd.DataFrame, X_val: pd.DataFrame,
                                       X_test: pd.DataFrame, y_train: pd.DataFrame,
                                       y_val: pd.DataFrame, y_test: pd.DataFrame,
                                       save_dir: str = 'models') -> Dict:
    """
    Train and evaluate baseline models

    Args:
        X_train, X_val, X_test: Feature datasets
        y_train, y_val, y_test: Target datasets
        save_dir: Directory to save models

    Returns:
        Dictionary with all results
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    results = {}

    # Model configurations
    models_config = {
        'LinearRegression': {'model_type': 'linear'},
        'RandomForest': {
            'model_type': 'rf',
            'n_estimators': 200,
            'max_depth': 15,
            'min_samples_split': 5,
            'min_samples_leaf': 2
        }
    }

    print("\n" + "=" * 80)
    print(" TRAINING BASELINE MODELS")
    print("=" * 80)

    for model_name, config in models_config.items():
        print(f"\n{'='*80}")
        print(f" {model_name}")
        print(f"{'='*80}")

        # Create and train model
        model = BaselineModel(**config)
        model.fit(X_train, y_train, scale=True)

        # Evaluate on train, val, test
        print("\nEvaluating on training set...")
        train_metrics = model.evaluate(X_train, y_train)

        print("Evaluating on validation set...")
        val_metrics = model.evaluate(X_val, y_val)

        print("Evaluating on test set...")
        test_metrics = model.evaluate(X_test, y_test)

        # Save model
        model_path = save_path / f"{model_name.lower()}_model.pkl"
        model.save(str(model_path))

        # Store results
        results[model_name] = {
            'train': train_metrics,
            'val': val_metrics,
            'test': test_metrics
        }

        # Print summary
        print("\n" + "-" * 80)
        print(f" {model_name} - Test Set Performance")
        print("-" * 80)
        for target, metrics in test_metrics.items():
            print(f"\n{target}:")
            for metric_name, value in metrics.items():
                if value is not None:
                    print(f"  {metric_name:12s}: {value:>10.4f}")
                else:
                    print(f"  {metric_name:12s}: {'N/A':>10s}")

    return results


def save_results_to_csv(results: Dict, filepath: str = 'results/baseline_comparison.csv'):
    """
    Save results to CSV file

    Args:
        results: Results dictionary
        filepath: Output filepath
    """
    rows = []

    for model_name, datasets in results.items():
        for dataset_name, targets in datasets.items():
            for target_name, metrics in targets.items():
                row = {
                    'Model': model_name,
                    'Dataset': dataset_name,
                    'Target': target_name,
                    **metrics
                }
                rows.append(row)

    df = pd.DataFrame(rows)

    # Create directory if needed
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(filepath, index=False)
    print(f"\n✓ Results saved to: {filepath}")

    return df


if __name__ == "__main__":
    pass
