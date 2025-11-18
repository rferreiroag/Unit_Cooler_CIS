"""
Advanced baseline models for Unit Cooler Digital Twin
Includes XGBoost, LightGBM, and MLP implementations
"""

import numpy as np
import pandas as pd
import time
from typing import Dict, Tuple, List, Optional
from pathlib import Path
import joblib
import json

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')


class AdvancedBaselineModel:
    """Wrapper for advanced baseline models"""

    def __init__(self, model_type: str = 'xgboost', **kwargs):
        """
        Initialize advanced model

        Args:
            model_type: 'xgboost', 'lightgbm', or 'mlp'
            **kwargs: Model-specific parameters
        """
        self.model_type = model_type
        self.model = None
        self.models_per_target = {}  # For multi-output
        self.feature_names = None
        self.target_names = None
        self.is_fitted = False
        self.training_time = 0
        self.params = kwargs

    def _create_xgboost_model(self, **kwargs):
        """Create XGBoost model"""
        try:
            from xgboost import XGBRegressor
        except ImportError:
            raise ImportError("XGBoost not installed. Run: pip install xgboost")

        default_params = {
            'n_estimators': 300,
            'max_depth': 8,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 3,
            'gamma': 0.1,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'random_state': 42,
            'n_jobs': -1,
            'tree_method': 'hist',
            'early_stopping_rounds': 50
        }
        default_params.update(kwargs)

        return XGBRegressor(**default_params)

    def _create_lightgbm_model(self, **kwargs):
        """Create LightGBM model"""
        try:
            from lightgbm import LGBMRegressor
        except ImportError:
            raise ImportError("LightGBM not installed. Run: pip install lightgbm")

        default_params = {
            'n_estimators': 300,
            'max_depth': 8,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 3,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1
        }
        default_params.update(kwargs)

        return LGBMRegressor(**default_params)

    def _create_mlp_model(self, n_features: int, n_targets: int, **kwargs):
        """Create MLP (Multi-Layer Perceptron) model"""
        try:
            import tensorflow as tf
        except ImportError:
            raise ImportError("TensorFlow not installed. Run: pip install tensorflow")

        # Default architecture
        hidden_layers = kwargs.get('hidden_layers', [128, 64, 32])
        dropout = kwargs.get('dropout', 0.3)
        learning_rate = kwargs.get('learning_rate', 0.001)

        model = tf.keras.Sequential()

        # Input layer
        model.add(tf.keras.layers.Input(shape=(n_features,)))

        # Hidden layers
        for i, units in enumerate(hidden_layers):
            model.add(tf.keras.layers.Dense(units, activation='relu', name=f'dense_{i+1}'))
            if dropout > 0:
                model.add(tf.keras.layers.Dropout(dropout, name=f'dropout_{i+1}'))

        # Output layer
        model.add(tf.keras.layers.Dense(n_targets, name='output'))

        # Compile
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

        return model

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
            feature_names: Optional[List[str]] = None,
            target_names: Optional[List[str]] = None):
        """
        Train the model

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            feature_names: Feature column names
            target_names: Target column names
        """
        self.feature_names = feature_names or [f'feature_{i}' for i in range(X_train.shape[1])]
        self.target_names = target_names or [f'target_{i}' for i in range(y_train.shape[1])]

        n_features = X_train.shape[1]
        n_targets = y_train.shape[1] if len(y_train.shape) > 1 else 1

        print(f"\nTraining {self.model_type.upper()} model...")
        print(f"  Input shape: {X_train.shape}")
        print(f"  Output shape: {y_train.shape}")
        print(f"  Targets: {self.target_names}")

        start_time = time.time()

        if self.model_type == 'mlp':
            # MLP handles multi-output natively
            self.model = self._create_mlp_model(n_features, n_targets, **self.params)

            # Training callbacks
            import tensorflow as tf
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss' if X_val is not None else 'loss',
                    patience=20,
                    restore_best_weights=True
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss' if X_val is not None else 'loss',
                    factor=0.5,
                    patience=10,
                    min_lr=1e-6
                )
            ]

            # Train
            validation_data = (X_val, y_val) if X_val is not None and y_val is not None else None

            history = self.model.fit(
                X_train, y_train,
                validation_data=validation_data,
                epochs=self.params.get('epochs', 200),
                batch_size=self.params.get('batch_size', 32),
                callbacks=callbacks,
                verbose=0
            )

            self.training_history = history.history

        else:
            # XGBoost and LightGBM - train separate model for each target
            for i, target_name in enumerate(self.target_names):
                print(f"  Training for {target_name}...")

                if self.model_type == 'xgboost':
                    model = self._create_xgboost_model(**self.params)
                elif self.model_type == 'lightgbm':
                    model = self._create_lightgbm_model(**self.params)
                else:
                    raise ValueError(f"Unknown model type: {self.model_type}")

                # Extract single target
                y_train_i = y_train[:, i] if len(y_train.shape) > 1 else y_train

                # Train with validation if provided
                if X_val is not None and y_val is not None:
                    y_val_i = y_val[:, i] if len(y_val.shape) > 1 else y_val

                    if self.model_type == 'xgboost':
                        model.fit(
                            X_train, y_train_i,
                            eval_set=[(X_val, y_val_i)],
                            verbose=False
                        )
                    else:  # lightgbm
                        model.fit(
                            X_train, y_train_i,
                            eval_set=[(X_val, y_val_i)],
                            callbacks=[
                                # LightGBM callback for early stopping
                            ]
                        )
                else:
                    model.fit(X_train, y_train_i)

                self.models_per_target[target_name] = model

        self.training_time = time.time() - start_time
        self.is_fitted = True

        print(f"  ✓ Training complete in {self.training_time:.2f}s")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions

        Args:
            X: Input features

        Returns:
            Predictions array
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")

        if self.model_type == 'mlp':
            return self.model.predict(X, verbose=0)
        else:
            # Combine predictions from all target models
            predictions = []
            for target_name in self.target_names:
                pred = self.models_per_target[target_name].predict(X)
                predictions.append(pred)

            return np.column_stack(predictions)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict:
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
            y_true_i = y[:, i] if len(y.shape) > 1 else y
            y_pred_i = y_pred[:, i] if len(y_pred.shape) > 1 else y_pred

            # Remove NaN values
            mask = ~np.isnan(y_true_i) & ~np.isnan(y_pred_i)
            y_true_clean = y_true_i[mask]
            y_pred_clean = y_pred_i[mask]

            if len(y_true_clean) > 0:
                mae = mean_absolute_error(y_true_clean, y_pred_clean)
                rmse = np.sqrt(mean_squared_error(y_true_clean, y_pred_clean))
                r2 = r2_score(y_true_clean, y_pred_clean)

                # MAPE
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

    def get_feature_importance(self, method: str = 'gain') -> pd.DataFrame:
        """
        Get feature importance

        Args:
            method: Importance type ('gain', 'weight', 'cover' for tree models)

        Returns:
            DataFrame with feature importances
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")

        if self.model_type == 'mlp':
            print("Feature importance not directly available for MLP")
            return None

        importance_dict = {}

        for target_name, model in self.models_per_target.items():
            if hasattr(model, 'feature_importances_'):
                importance_dict[target_name] = model.feature_importances_
            else:
                print(f"Feature importance not available for {target_name}")

        if importance_dict:
            importance_df = pd.DataFrame(importance_dict, index=self.feature_names)
            importance_df['mean'] = importance_df.mean(axis=1)
            importance_df = importance_df.sort_values('mean', ascending=False)
            return importance_df

        return None

    def save(self, filepath: str):
        """Save model to disk"""
        if self.model_type == 'mlp':
            # Save Keras model
            self.model.save(filepath)
            # Save metadata separately
            metadata = {
                'model_type': self.model_type,
                'feature_names': self.feature_names,
                'target_names': self.target_names,
                'training_time': self.training_time,
                'params': self.params
            }
            with open(filepath + '.meta.json', 'w') as f:
                json.dump(metadata, f, indent=2)
        else:
            # Save tree models with joblib
            joblib.dump({
                'models': self.models_per_target,
                'model_type': self.model_type,
                'feature_names': self.feature_names,
                'target_names': self.target_names,
                'training_time': self.training_time,
                'params': self.params
            }, filepath)

        print(f"✓ Model saved: {filepath}")


def temporal_cross_validation(X: np.ndarray, y: np.ndarray,
                              model_type: str,
                              n_splits: int = 5,
                              **model_params) -> Dict:
    """
    Perform temporal cross-validation

    Args:
        X: Features
        y: Targets
        model_type: Type of model ('xgboost', 'lightgbm', 'mlp')
        n_splits: Number of CV splits
        **model_params: Model parameters

    Returns:
        Dictionary with CV results
    """
    print(f"\n{'='*80}")
    print(f" TEMPORAL CROSS-VALIDATION: {model_type.upper()}")
    print(f"{'='*80}")
    print(f"\nFolds: {n_splits}")

    tscv = TimeSeriesSplit(n_splits=n_splits)

    cv_results = {
        'train_scores': [],
        'val_scores': [],
        'fold_metrics': []
    }

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
        print(f"\nFold {fold}/{n_splits}")
        print(f"  Train: {len(train_idx):,} samples")
        print(f"  Val:   {len(val_idx):,} samples")

        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]

        # Train model
        model = AdvancedBaselineModel(model_type=model_type, **model_params)
        model.fit(X_train_fold, y_train_fold, X_val_fold, y_val_fold)

        # Evaluate
        train_metrics = model.evaluate(X_train_fold, y_train_fold)
        val_metrics = model.evaluate(X_val_fold, y_val_fold)

        cv_results['train_scores'].append(train_metrics)
        cv_results['val_scores'].append(val_metrics)

        # Print fold results
        for target in model.target_names:
            val_r2 = val_metrics[target]['R2']
            val_mae = val_metrics[target]['MAE']
            print(f"    {target}: R²={val_r2:.4f}, MAE={val_mae:.4f}")

    # Aggregate results
    print(f"\n{'='*80}")
    print(" CROSS-VALIDATION SUMMARY")
    print(f"{'='*80}\n")

    # Calculate mean and std for each metric
    for target in model.target_names:
        val_r2_scores = [fold[target]['R2'] for fold in cv_results['val_scores']]
        val_mae_scores = [fold[target]['MAE'] for fold in cv_results['val_scores']]

        print(f"{target}:")
        print(f"  R²:  {np.mean(val_r2_scores):.4f} ± {np.std(val_r2_scores):.4f}")
        print(f"  MAE: {np.mean(val_mae_scores):.4f} ± {np.std(val_mae_scores):.4f}")

    return cv_results


if __name__ == "__main__":
    # Test with processed data
    import sys
    sys.path.append('../..')

    # Load processed data
    X_train = np.load('../../data/processed/X_train_scaled.npy')
    y_train = np.load('../../data/processed/y_train_scaled.npy')
    X_val = np.load('../../data/processed/X_val_scaled.npy')
    y_val = np.load('../../data/processed/y_val_scaled.npy')

    # Load metadata
    with open('../../data/processed/metadata.json', 'r') as f:
        metadata = json.load(f)

    feature_names = metadata['feature_names']
    target_names = metadata['target_names']

    print(f"Loaded data: X_train={X_train.shape}, y_train={y_train.shape}")

    # Test XGBoost
    model = AdvancedBaselineModel(model_type='xgboost')
    model.fit(X_train, y_train, X_val, y_val,
             feature_names=feature_names, target_names=target_names)

    metrics = model.evaluate(X_val, y_val)
    print("\nValidation metrics:")
    for target, m in metrics.items():
        print(f"  {target}: R²={m['R2']:.4f}, MAE={m['MAE']:.4f}")
