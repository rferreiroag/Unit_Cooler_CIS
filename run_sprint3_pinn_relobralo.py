"""
Sprint 3: Physics-Informed Neural Network with ReLoBRaLo
=========================================================

This script implements the state-of-the-art ReLoBRaLo (Relative Loss
Balancing with Random Lookback) algorithm for adaptive loss balancing
in Physics-Informed Neural Networks.

Reference:
    Bischof, R., & Kraus, M. (2025). Multi-Objective Loss Balancing for
    Physics-Informed Deep Learning. Computer Methods in Applied Mechanics
    and Engineering.

    GitHub: https://github.com/rbischof/relative_balancing

Usage:
    python run_sprint3_pinn_relobralo.py
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
from tensorflow import keras

from models.pinn_model import create_pinn_model, PINN
from models.advanced_models import AdvancedBaselineModel

import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)


class ReLoBRaLoCallback(keras.callbacks.Callback):
    """
    ReLoBRaLo: Relative Loss Balancing with Random Lookback

    This callback implements adaptive loss balancing for PINNs using:
    1. Relative loss progress (L_j(i) / L_j(i'))
    2. Softmax with temperature parameter
    3. Exponential moving average
    4. Random lookback mechanism

    Mathematical formulation:
        w_j^(i;i') = n_loss * exp(L_j(i) / (tau * L_j(i'))) / sum(...)
        w_j(i) = alpha * w_j(i-1) + (1-alpha) * w_j^(i;i')

    Args:
        n_loss_terms: Number of loss terms (data loss + physics loss)
        temperature: Temperature parameter tau for softmax (default: 1.0)
        alpha: Exponential moving average parameter (default: 0.999)
        lookback_range: Range for random lookback (default: (1, 10))
        update_freq: Update weights every N epochs (default: 1)
    """

    def __init__(self,
                 n_loss_terms: int = 2,
                 temperature: float = 1.0,
                 alpha: float = 0.999,
                 lookback_range: tuple = (1, 10),
                 update_freq: int = 1):
        super().__init__()
        self.n_loss_terms = n_loss_terms
        self.temperature = temperature
        self.alpha = alpha
        self.lookback_min, self.lookback_max = lookback_range
        self.update_freq = update_freq

        # Loss history storage
        self.loss_history = {
            'data_loss': [],
            'physics_loss': []
        }

        # Weight history
        self.weight_history = {
            'lambda_data': [],
            'lambda_physics': []
        }

        # Current weights (moving average)
        self.current_weights = np.array([1.0, 0.0])  # Start with data-only

    def on_epoch_end(self, epoch, logs=None):
        """Update loss weights using ReLoBRaLo algorithm"""

        if logs is None:
            return

        # Extract current losses
        current_data_loss = logs.get('data_loss', 0.0)
        current_physics_loss = logs.get('physics_loss', 0.0)

        # Store in history
        self.loss_history['data_loss'].append(current_data_loss)
        self.loss_history['physics_loss'].append(current_physics_loss)

        # Skip first epoch (no history yet)
        if epoch == 0:
            self.weight_history['lambda_data'].append(self.current_weights[0])
            self.weight_history['lambda_physics'].append(self.current_weights[1])
            return

        # Update weights only at specified frequency
        if (epoch + 1) % self.update_freq != 0:
            return

        # Random lookback: select i' from past iterations
        max_lookback = min(epoch, self.lookback_max)
        lookback_idx = np.random.randint(
            max(0, epoch - max_lookback),
            max(1, epoch - self.lookback_min + 1)
        )

        # Current losses L_j(i)
        current_losses = np.array([current_data_loss, current_physics_loss])

        # Past losses L_j(i')
        past_data_loss = self.loss_history['data_loss'][lookback_idx]
        past_physics_loss = self.loss_history['physics_loss'][lookback_idx]
        past_losses = np.array([past_data_loss, past_physics_loss])

        # Avoid division by zero
        past_losses = np.maximum(past_losses, 1e-10)

        # Relative progress: L_j(i) / L_j(i')
        relative_progress = current_losses / past_losses

        # Softmax with temperature
        # w_j = n_loss * exp(L_j(i) / (tau * L_j(i'))) / sum(...)
        scaled_progress = relative_progress / self.temperature
        exp_progress = np.exp(scaled_progress - np.max(scaled_progress))  # Numerical stability
        intermediate_weights = self.n_loss_terms * exp_progress / np.sum(exp_progress)

        # Exponential moving average
        # w_j(i) = alpha * w_j(i-1) + (1-alpha) * w_j^(i;i')
        self.current_weights = (
            self.alpha * self.current_weights +
            (1 - self.alpha) * intermediate_weights
        )

        # Ensure non-negative and normalized
        self.current_weights = np.maximum(self.current_weights, 0.0)
        weight_sum = np.sum(self.current_weights)
        if weight_sum > 0:
            self.current_weights = self.current_weights / weight_sum * self.n_loss_terms

        # Update model weights
        self.model.lambda_data = float(self.current_weights[0])
        self.model.lambda_physics = float(self.current_weights[1])

        # Store in history
        self.weight_history['lambda_data'].append(self.model.lambda_data)
        self.weight_history['lambda_physics'].append(self.model.lambda_physics)

        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"\n  [ReLoBRaLo] Epoch {epoch+1}:")
            print(f"    λ_data    = {self.model.lambda_data:.4f}")
            print(f"    λ_physics = {self.model.lambda_physics:.4f}")
            print(f"    L_data    = {current_data_loss:.6f}")
            print(f"    L_physics = {current_physics_loss:.6f}")
            print(f"    Lookback  = {epoch - lookback_idx} epochs")


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


def train_pinn_relobralo(data_dict, config: dict):
    """
    Train PINN with ReLoBRaLo adaptive loss balancing

    Args:
        data_dict: Dictionary with data splits
        config: PINN configuration

    Returns:
        Trained model, history, and ReLoBRaLo callback
    """
    print("\n" + "="*80)
    print(" TRAINING PINN WITH RELOBRALO")
    print(" State-of-the-Art Adaptive Loss Balancing (2024-2025)")
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

    # Create model with initial weights
    # Start with data-focused training (physics will be added adaptively)
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
        lambda_data=1.0,
        lambda_physics=0.0  # Will be adjusted by ReLoBRaLo
    )

    # ReLoBRaLo callback
    relobralo_callback = ReLoBRaLoCallback(
        n_loss_terms=2,  # data_loss + physics_loss
        temperature=config.get('temperature', 1.0),
        alpha=config.get('alpha', 0.999),
        lookback_range=config.get('lookback_range', (1, 10)),
        update_freq=config.get('update_freq', 1)
    )

    # Other callbacks
    callbacks = [
        relobralo_callback,
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=50,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=20,
            min_lr=1e-6,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            'models/pinn_relobralo_best_weights.weights.h5',
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=True,
            verbose=0
        )
    ]

    # Train
    print("\n" + "-"*80)
    print(" ReLoBRaLo Configuration:")
    print("-"*80)
    print(f"  Temperature (τ):      {config.get('temperature', 1.0)}")
    print(f"  Moving Average (α):   {config.get('alpha', 0.999)}")
    print(f"  Lookback Range:       {config.get('lookback_range', (1, 10))}")
    print(f"  Update Frequency:     {config.get('update_freq', 1)} epochs")
    print(f"  Total Epochs:         {config.get('epochs', 200)}")
    print(f"  Batch Size:           {config.get('batch_size', 64)}")
    print("-"*80)

    print("\nStarting training with adaptive loss balancing...")
    print("ReLoBRaLo will automatically adjust λ_data and λ_physics")

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=config.get('epochs', 200),
        batch_size=config.get('batch_size', 64),
        callbacks=callbacks,
        verbose=1
    )

    print("\n✓ Training complete")

    return model, history, relobralo_callback


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


def compare_with_baselines(pinn_results, target_names, approach_name="PINN-ReLoBRaLo"):
    """Compare PINN with baseline models"""
    print("\n" + "="*80)
    print(f" COMPARING {approach_name} VS BASELINES")
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
            'Model': approach_name,
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
    output_file = 'results/pinn_relobralo_vs_baselines.csv'
    comp_df.to_csv(output_file, index=False)
    print(f"\n✓ Comparison saved to: {output_file}")

    return comp_df


def plot_relobralo_history(history, relobralo_callback):
    """Plot training history with ReLoBRaLo weight evolution"""
    print("\nGenerating ReLoBRaLo training history plots...")

    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # Row 1: Loss evolution
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(history.history['loss'], label='Train', linewidth=2)
    ax1.plot(history.history['val_loss'], label='Validation', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('Total Loss', fontsize=11)
    ax1.set_title('Total Loss (Data + Physics)', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Data loss
    ax2 = fig.add_subplot(gs[0, 1])
    if 'data_loss' in history.history:
        ax2.plot(history.history['data_loss'], label='Train Data Loss', linewidth=2)
        ax2.plot(history.history['val_data_loss'], label='Val Data Loss', linewidth=2)
        ax2.set_xlabel('Epoch', fontsize=11)
        ax2.set_ylabel('Data Loss (MSE)', fontsize=11)
        ax2.set_title('Data-Driven Loss', fontsize=12, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)

    # Physics loss
    ax3 = fig.add_subplot(gs[0, 2])
    if 'physics_loss' in history.history:
        ax3.plot(history.history['physics_loss'], label='Train Physics Loss', linewidth=2, color='coral')
        ax3.plot(history.history['val_physics_loss'], label='Val Physics Loss', linewidth=2, color='orange')
        ax3.set_xlabel('Epoch', fontsize=11)
        ax3.set_ylabel('Physics Loss', fontsize=11)
        ax3.set_title('Physics-Informed Loss', fontsize=12, fontweight='bold')
        ax3.legend(fontsize=10)
        ax3.grid(True, alpha=0.3)

    # Row 2: ReLoBRaLo weight evolution
    ax4 = fig.add_subplot(gs[1, :])
    epochs = range(len(relobralo_callback.weight_history['lambda_data']))
    ax4.plot(epochs, relobralo_callback.weight_history['lambda_data'],
             label='λ_data', linewidth=2, color='blue')
    ax4.plot(epochs, relobralo_callback.weight_history['lambda_physics'],
             label='λ_physics', linewidth=2, color='red')
    ax4.set_xlabel('Epoch', fontsize=11)
    ax4.set_ylabel('Loss Weight', fontsize=11)
    ax4.set_title('ReLoBRaLo Adaptive Weight Evolution', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)

    # Row 3: Loss history (for relative progress visualization)
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.plot(relobralo_callback.loss_history['data_loss'],
             label='Data Loss', linewidth=2, color='blue', alpha=0.7)
    ax5.set_xlabel('Epoch', fontsize=11)
    ax5.set_ylabel('Data Loss Value', fontsize=11)
    ax5.set_title('Data Loss History (for ReLoBRaLo)', fontsize=12, fontweight='bold')
    ax5.set_yscale('log')
    ax5.legend(fontsize=10)
    ax5.grid(True, alpha=0.3)

    ax6 = fig.add_subplot(gs[2, 1])
    ax6.plot(relobralo_callback.loss_history['physics_loss'],
             label='Physics Loss', linewidth=2, color='red', alpha=0.7)
    ax6.set_xlabel('Epoch', fontsize=11)
    ax6.set_ylabel('Physics Loss Value', fontsize=11)
    ax6.set_title('Physics Loss History (for ReLoBRaLo)', fontsize=12, fontweight='bold')
    ax6.set_yscale('log')
    ax6.legend(fontsize=10)
    ax6.grid(True, alpha=0.3)

    # Weight ratio
    ax7 = fig.add_subplot(gs[2, 2])
    lambda_data = np.array(relobralo_callback.weight_history['lambda_data'])
    lambda_physics = np.array(relobralo_callback.weight_history['lambda_physics'])
    weight_ratio = lambda_physics / (lambda_data + 1e-10)
    ax7.plot(epochs, weight_ratio, linewidth=2, color='purple')
    ax7.set_xlabel('Epoch', fontsize=11)
    ax7.set_ylabel('λ_physics / λ_data', fontsize=11)
    ax7.set_title('Physics-to-Data Weight Ratio', fontsize=12, fontweight='bold')
    ax7.set_yscale('log')
    ax7.grid(True, alpha=0.3)

    plt.suptitle('PINN Training with ReLoBRaLo (State-of-the-Art 2024-2025)',
                 fontsize=14, fontweight='bold', y=0.995)

    output_dir = Path('plots/sprint3')
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.savefig(output_dir / 'pinn_relobralo_training_history.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: plots/sprint3/pinn_relobralo_training_history.png")
    plt.close()


def main():
    """Execute complete Sprint 3 pipeline with ReLoBRaLo"""

    print("\n" + "="*80)
    print(" SPRINT 3: PINN WITH RELOBRALO")
    print(" State-of-the-Art Adaptive Loss Balancing (2024-2025)")
    print("="*80)
    print("\n Reference:")
    print("   Bischof, R., & Kraus, M. (2025). Multi-Objective Loss Balancing")
    print("   for Physics-Informed Deep Learning. Computer Methods in Applied")
    print("   Mechanics and Engineering.")
    print("="*80)

    # Load data
    data_dict = load_processed_data()

    # PINN with ReLoBRaLo configuration
    pinn_config = {
        'hidden_layers': [128, 128, 64, 32],
        'dropout': 0.2,
        # ReLoBRaLo parameters
        'temperature': 1.0,      # τ: softmax temperature
        'alpha': 0.999,          # α: exponential moving average
        'lookback_range': (1, 10),  # Random lookback range
        'update_freq': 1,        # Update weights every epoch
        # Training parameters
        'epochs': 200,
        'batch_size': 64
    }

    print("\n" + "-"*80)
    print(" Configuration Summary:")
    print("-"*80)
    print(" Model Architecture:")
    for key in ['hidden_layers', 'dropout']:
        print(f"   {key}: {pinn_config[key]}")
    print("\n ReLoBRaLo Hyperparameters:")
    for key in ['temperature', 'alpha', 'lookback_range', 'update_freq']:
        print(f"   {key}: {pinn_config[key]}")
    print("\n Training Parameters:")
    for key in ['epochs', 'batch_size']:
        print(f"   {key}: {pinn_config[key]}")
    print("-"*80)

    # Train PINN with ReLoBRaLo
    model, history, relobralo_callback = train_pinn_relobralo(data_dict, pinn_config)

    # Plot training history
    plot_relobralo_history(history, relobralo_callback)

    # Evaluate on test set
    pinn_results, y_pred = evaluate_pinn(
        model,
        data_dict['X_test'],
        data_dict['y_test'],
        data_dict['target_names']
    )

    # Compare with baselines
    comparison_df = compare_with_baselines(
        pinn_results,
        data_dict['target_names'],
        approach_name="PINN-ReLoBRaLo"
    )

    # Save model
    try:
        model.save('models/pinn_relobralo_model.keras')
        print(f"\n✓ Model saved: models/pinn_relobralo_model.keras")
    except Exception as e:
        print(f"\nWarning: Full model save failed: {e}")
        print("Saving weights only...")
        model.save_weights('models/pinn_relobralo_weights.h5')
        print(f"✓ Model weights saved: models/pinn_relobralo_weights.h5")

    # Final summary
    print("\n" + "="*80)
    print(" SPRINT 3: RELOBRALO RESULTS")
    print("="*80)

    print(f"""
┌─────────────────────────────────────────────────────────────────┐
│              PINN-ReLoBRaLo PERFORMANCE (Test Set)             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Final Adaptive Weights:                                       │
│    λ_data    = {relobralo_callback.current_weights[0]:.4f}                                          │
│    λ_physics = {relobralo_callback.current_weights[1]:.4f}                                          │
│                                                                 │
│  Model Performance:                                            │""")

    for target in data_dict['target_names']:
        metrics = pinn_results[target]
        print(f"│    {target:10s}: R²={metrics['R2']:.4f}, MAE={metrics['MAE']:.4f}, MAPE={metrics['MAPE']:.2f}%   │")

    print(f"""│                                                                 │
│  Output Files:                                                  │
│    • models/pinn_relobralo_model.keras                         │
│    • results/pinn_relobralo_vs_baselines.csv                   │
│    • plots/sprint3/pinn_relobralo_training_history.png         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
""")

    # Comparison with LightGBM baseline
    print("\n" + "="*80)
    print(" FINAL COMPARISON: PINN-ReLoBRaLo vs LightGBM")
    print("="*80)

    for target in data_dict['target_names']:
        pinn_r2 = pinn_results[target]['R2']
        lgbm_row = comparison_df[
            (comparison_df['Target'] == target) &
            (comparison_df['Model'] == 'LightGBM')
        ].iloc[0]
        lgbm_r2 = lgbm_row['R2']

        improvement = ((pinn_r2 - lgbm_r2) / abs(lgbm_r2)) * 100 if lgbm_r2 != 0 else 0

        print(f"\n{target}:")
        print(f"  PINN-ReLoBRaLo: R²={pinn_r2:.4f}")
        print(f"  LightGBM:       R²={lgbm_r2:.4f}")
        if improvement > 0:
            print(f"  ✓ PINN improved by {improvement:.2f}%")
        else:
            print(f"  ✗ LightGBM better by {-improvement:.2f}%")

    print("\n" + "="*80)


if __name__ == "__main__":
    main()
