"""
Sprint 3: Physics-Informed Neural Network (PINN)

This script trains a PINN that combines data-driven learning with
thermodynamic constraints for the Unit Cooler digital twin.

Usage:
    python run_sprint3_pinn.py
"""

import sys
sys.path.append('src')

import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import tensorflow as tf

from models.pinn_model import create_pinn_model, PINN
from models.advanced_models import AdvancedBaselineModel

import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)


def load_processed_data():
    """Load processed data from Sprint 1"""
    print("\n" + "="*80)
    print(" LOADING PROCESSED DATA")
    print("="*80)

    data_dir = Path('data/processed')

    # Load scaled arrays
    X_train = np.load(data_dir / 'X_train_scaled.npy').astype(np.float32)
    y_train = np.load(data_dir / 'y_train_scaled.npy').astype(np.float32)
    X_val = np.load(data_dir / 'X_val_scaled.npy').astype(np.float32)
    y_val = np.load(data_dir / 'y_val_scaled.npy').astype(np.float32)
    X_test = np.load(data_dir / 'X_test_scaled.npy').astype(np.float32)
    y_test = np.load(data_dir / 'y_test_scaled.npy').astype(np.float32)

    # Load scaler parameters for physics loss unscaling
    X_mean = np.load(data_dir / 'X_scaler_mean.npy').astype(np.float32)
    X_scale = np.load(data_dir / 'X_scaler_scale.npy').astype(np.float32)
    y_mean = np.load(data_dir / 'y_scaler_mean.npy').astype(np.float32)
    y_scale = np.load(data_dir / 'y_scaler_scale.npy').astype(np.float32)

    # Load metadata
    with open(data_dir / 'metadata.json', 'r') as f:
        metadata = json.load(f)

    print(f"\n✓ Data loaded successfully")
    print(f"  Train: {X_train.shape}")
    print(f"  Val:   {X_val.shape}")
    print(f"  Test:  {X_test.shape}")
    print(f"  Features: {len(metadata['feature_names'])}")
    print(f"  Targets: {metadata['target_names']}")
    print(f"  Scaler parameters loaded for physics loss unscaling")

    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val,
        'X_test': X_test,
        'y_test': y_test,
        'X_mean': X_mean,
        'X_scale': X_scale,
        'y_mean': y_mean,
        'y_scale': y_scale,
        'feature_names': metadata['feature_names'],
        'target_names': metadata['target_names']
    }


def train_pinn(data_dict, config: dict):
    """
    Train PINN model

    Args:
        data_dict: Dictionary with data splits
        config: PINN configuration

    Returns:
        Trained model and history
    """
    print("\n" + "="*80)
    print(" TRAINING PHYSICS-INFORMED NEURAL NETWORK")
    print("="*80)

    X_train = data_dict['X_train']
    y_train = data_dict['y_train']
    X_val = data_dict['X_val']
    y_val = data_dict['y_val']
    feature_names = data_dict['feature_names']
    target_names = data_dict['target_names']

    # Scaler parameters for physics loss
    X_mean = data_dict['X_mean']
    X_scale = data_dict['X_scale']
    y_mean = data_dict['y_mean']
    y_scale = data_dict['y_scale']

    n_features = X_train.shape[1]
    n_targets = y_train.shape[1]

    # Create model with scaler parameters
    model = create_pinn_model(
        n_features=n_features,
        feature_names=feature_names,
        n_targets=n_targets,
        hidden_layers=config.get('hidden_layers', [128, 128, 64, 32]),
        dropout=config.get('dropout', 0.2),
        X_mean=X_mean,
        X_scale=X_scale,
        y_mean=y_mean,
        y_scale=y_scale,
        lambda_data=config.get('lambda_data', 1.0),
        lambda_physics=config.get('lambda_physics', 0.1)
    )

    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=30,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=15,
            min_lr=1e-6,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            'models/pinn_best_weights.weights.h5',
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=True,
            verbose=0
        )
    ]

    # Train
    print("\nStarting training...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=config.get('epochs', 200),
        batch_size=config.get('batch_size', 64),
        callbacks=callbacks,
        verbose=1
    )

    print("\n✓ Training complete")

    return model, history


def evaluate_pinn(model, X_test, y_test, target_names):
    """Evaluate PINN on test set"""
    print("\n" + "="*80)
    print(" EVALUATING PINN ON TEST SET")
    print("="*80)

    # Predictions
    y_pred = model.predict(X_test, verbose=0)

    # Calculate metrics
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    results = {}
    for i, target in enumerate(target_names):
        y_true = y_test[:, i]
        y_pred_i = y_pred[:, i]

        mae = mean_absolute_error(y_true, y_pred_i)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred_i))
        r2 = r2_score(y_true, y_pred_i)

        # MAPE
        mape_mask = y_true != 0
        if mape_mask.sum() > 0:
            mape = np.mean(np.abs((y_true[mape_mask] - y_pred_i[mape_mask]) / y_true[mape_mask])) * 100
        else:
            mape = np.nan

        results[target] = {
            'MAE': round(mae, 4),
            'RMSE': round(rmse, 4),
            'MAPE': round(mape, 4) if not np.isnan(mape) else None,
            'R2': round(r2, 4)
        }

        print(f"\n{target}:")
        print(f"  MAE:  {mae:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAPE: {mape:.2f}%")
        print(f"  R²:   {r2:.4f}")

    return results, y_pred


def compare_with_baselines(pinn_results, target_names):
    """Compare PINN with baseline models"""
    print("\n" + "="*80)
    print(" COMPARING PINN VS BASELINES")
    print("="*80)

    # Load baseline results
    baseline_df = pd.read_csv('results/advanced_baseline_comparison.csv')
    test_df = baseline_df[baseline_df['Dataset'] == 'test']

    comparison = []

    for target in target_names:
        # PINN
        pinn_metrics = pinn_results[target]

        # Best baseline (LightGBM)
        lgbm = test_df[(test_df['Model'] == 'LightGBM') & (test_df['Target'] == target)].iloc[0]

        # XGBoost
        xgb = test_df[(test_df['Model'] == 'XGBoost') & (test_df['Target'] == target)].iloc[0]

        comparison.append({
            'Target': target,
            'Model': 'PINN',
            'R2': pinn_metrics['R2'],
            'MAE': pinn_metrics['MAE'],
            'MAPE': pinn_metrics['MAPE']
        })

        comparison.append({
            'Target': target,
            'Model': 'LightGBM',
            'R2': lgbm['R2'],
            'MAE': lgbm['MAE'],
            'MAPE': lgbm['MAPE']
        })

        comparison.append({
            'Target': target,
            'Model': 'XGBoost',
            'R2': xgb['R2'],
            'MAE': xgb['MAE'],
            'MAPE': xgb['MAPE']
        })

    comp_df = pd.DataFrame(comparison)

    print("\nModel Comparison (Test Set):")
    print("="*80)

    for target in target_names:
        print(f"\n{target}:")
        target_df = comp_df[comp_df['Target'] == target][['Model', 'R2', 'MAE', 'MAPE']]
        target_df = target_df.sort_values('R2', ascending=False)
        print(target_df.to_string(index=False))

    # Save comparison
    comp_df.to_csv('results/pinn_vs_baselines.csv', index=False)
    print(f"\n✓ Comparison saved to: results/pinn_vs_baselines.csv")

    return comp_df


def plot_training_history(history):
    """Plot training history"""
    print("\nGenerating training history plots...")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Total loss
    axes[0].plot(history.history['loss'], label='Train', linewidth=2)
    axes[0].plot(history.history['val_loss'], label='Validation', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=11)
    axes[0].set_ylabel('Total Loss', fontsize=11)
    axes[0].set_title('Total Loss (Data + Physics)', fontsize=12, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # Data loss
    if 'data_loss' in history.history:
        axes[1].plot(history.history['data_loss'], label='Train Data Loss', linewidth=2)
        axes[1].plot(history.history['val_data_loss'], label='Val Data Loss', linewidth=2)
        axes[1].set_xlabel('Epoch', fontsize=11)
        axes[1].set_ylabel('Data Loss (MSE)', fontsize=11)
        axes[1].set_title('Data-Driven Loss', fontsize=12, fontweight='bold')
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)

    # Physics loss
    if 'physics_loss' in history.history:
        axes[2].plot(history.history['physics_loss'], label='Train Physics Loss', linewidth=2, color='coral')
        axes[2].plot(history.history['val_physics_loss'], label='Val Physics Loss', linewidth=2, color='orange')
        axes[2].set_xlabel('Epoch', fontsize=11)
        axes[2].set_ylabel('Physics Loss', fontsize=11)
        axes[2].set_title('Physics-Informed Loss', fontsize=12, fontweight='bold')
        axes[2].legend(fontsize=10)
        axes[2].grid(True, alpha=0.3)

    plt.suptitle('PINN Training History', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    output_dir = Path('plots/sprint3')
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.savefig(output_dir / 'pinn_training_history.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: plots/sprint3/pinn_training_history.png")
    plt.close()


def main():
    """Execute complete Sprint 3 pipeline"""

    print("\n" + "="*80)
    print(" SPRINT 3: PHYSICS-INFORMED NEURAL NETWORK (PINN)")
    print(" Physics-Informed Digital Twin for Unit Cooler HVAC")
    print("="*80)

    # Load data
    data_dict = load_processed_data()

    # PINN configuration
    # Reduced lambda_physics to 0.001 and added normalization in physics loss
    pinn_config = {
        'hidden_layers': [128, 128, 64, 32],
        'dropout': 0.2,
        'lambda_data': 1.0,
        'lambda_physics': 0.001,  # Reduced from 0.01 after normalization
        'epochs': 200,
        'batch_size': 64
    }

    print("\nPINN Configuration:")
    for key, value in pinn_config.items():
        print(f"  {key}: {value}")

    # Train PINN
    model, history = train_pinn(data_dict, pinn_config)

    # Plot training history
    plot_training_history(history)

    # Evaluate on test set
    pinn_results, y_pred = evaluate_pinn(
        model,
        data_dict['X_test'],
        data_dict['y_test'],
        data_dict['target_names']
    )

    # Compare with baselines
    comparison_df = compare_with_baselines(pinn_results, data_dict['target_names'])

    # Save model (using .keras format for custom models)
    try:
        model.save('models/pinn_model.keras')
        print(f"\n✓ Model saved: models/pinn_model.keras")
    except Exception as e:
        print(f"\nWarning: Full model save failed: {e}")
        print("Saving weights only...")
        model.save_weights('models/pinn_model_weights.h5')
        print(f"✓ Model weights saved: models/pinn_model_weights.h5")

    # Summary
    print("\n" + "="*80)
    print(" SPRINT 3 COMPLETE")
    print("="*80)

    print(f"""
┌─────────────────────────────────────────────────────────────────┐
│                      SPRINT 3 SUMMARY                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  PINN Model Performance (Test Set):                            │
""")

    for target in data_dict['target_names']:
        metrics = pinn_results[target]
        print(f"│    {target:10s}: R²={metrics['R2']:.4f}, MAE={metrics['MAE']:.4f}, MAPE={metrics['MAPE']:.2f}%   │")

    print(f"""│                                                                 │
│  Output Files:                                                  │
│    • models/pinn_model.h5                                       │
│    • results/pinn_vs_baselines.csv                              │
│    • plots/sprint3/pinn_training_history.png                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
""")

    print("="*80)
    print(" READY FOR SPRINT 4: HYPERPARAMETER OPTIMIZATION")
    print("="*80)


if __name__ == "__main__":
    main()
