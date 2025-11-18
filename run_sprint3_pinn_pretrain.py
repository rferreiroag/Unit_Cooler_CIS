"""
Sprint 3: PINN with Pre-training and Gradual Physics Loss
Strategy: Two-phase training to address scale mismatch

Phase 1 (Pre-training): Train pure MLP without physics loss (50 epochs)
Phase 2 (Fine-tuning): Gradually increase physics loss from 0 to target value (50 epochs)

This approach allows the model to learn data patterns first, then enforce
physical constraints gradually without overwhelming the gradients.
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


class GradualPhysicsLossCallback(keras.callbacks.Callback):
    """
    Callback to gradually increase physics loss weight during training

    Implements curriculum learning by starting with pure data-driven learning
    and gradually introducing physics constraints.
    """

    def __init__(self, start_lambda: float = 0.0,
                 target_lambda: float = 0.001,
                 warmup_epochs: int = 10,
                 total_epochs: int = 50):
        """
        Args:
            start_lambda: Initial physics loss weight
            target_lambda: Final physics loss weight
            warmup_epochs: Number of epochs to stay at start_lambda
            total_epochs: Total number of epochs for phase 2
        """
        super().__init__()
        self.start_lambda = start_lambda
        self.target_lambda = target_lambda
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.current_epoch = 0

    def on_epoch_begin(self, epoch, logs=None):
        """Update lambda_physics at the start of each epoch"""
        self.current_epoch = epoch

        if epoch < self.warmup_epochs:
            # Warmup period: keep at start value
            new_lambda = self.start_lambda
        else:
            # Linear increase from start to target
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            progress = min(progress, 1.0)  # Cap at 1.0
            new_lambda = self.start_lambda + progress * (self.target_lambda - self.start_lambda)

        # Update model's lambda_physics
        self.model.lambda_physics = new_lambda

        if epoch % 10 == 0:
            print(f"\n  Epoch {epoch}: lambda_physics = {new_lambda:.6f}")


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


def pretrain_mlp(data_dict, config: dict):
    """
    Phase 1: Pre-train pure MLP without physics loss

    This allows the model to learn basic input-output mappings
    from data without being constrained by physics.
    """
    print("\n" + "="*80)
    print(" PHASE 1: PRE-TRAINING (MLP WITHOUT PHYSICS LOSS)")
    print("="*80)

    X_train = data_dict['X_train']
    y_train = data_dict['y_train']
    X_val = data_dict['X_val']
    y_val = data_dict['y_val']
    feature_names = data_dict['feature_names']

    # Scaler parameters (needed for model creation)
    X_mean = data_dict['X_mean']
    X_scale = data_dict['X_scale']
    y_mean = data_dict['y_mean']
    y_scale = data_dict['y_scale']

    n_features = X_train.shape[1]
    n_targets = y_train.shape[1]

    # Create model with physics loss DISABLED (lambda_physics=0)
    print("\nCreating MLP model (physics loss disabled)...")
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
        lambda_physics=0.0  # NO PHYSICS LOSS
    )

    # Callbacks for pre-training
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-6,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            'models/pinn_pretrained.weights.h5',
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=True,
            verbose=0
        )
    ]

    # Pre-train
    print(f"\nPre-training for {config['pretrain_epochs']} epochs...")
    print("(Pure data-driven learning without physics constraints)\n")

    history_phase1 = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=config['pretrain_epochs'],
        batch_size=config['batch_size'],
        callbacks=callbacks,
        verbose=1
    )

    print("\n✓ Phase 1 (Pre-training) complete")
    print(f"  Final val_loss: {history_phase1.history['val_loss'][-1]:.4f}")

    return model, history_phase1


def finetune_with_physics(model, data_dict, config: dict, history_phase1):
    """
    Phase 2: Fine-tune with gradually increasing physics loss

    Uses curriculum learning to introduce physics constraints slowly,
    preventing gradient explosion.
    """
    print("\n" + "="*80)
    print(" PHASE 2: FINE-TUNING (GRADUAL PHYSICS LOSS)")
    print("="*80)

    X_train = data_dict['X_train']
    y_train = data_dict['y_train']
    X_val = data_dict['X_val']
    y_val = data_dict['y_val']

    target_lambda = config.get('target_lambda_physics', 0.001)
    finetune_epochs = config.get('finetune_epochs', 50)
    warmup_epochs = config.get('warmup_epochs', 10)

    print(f"\nFine-tuning configuration:")
    print(f"  Target lambda_physics: {target_lambda}")
    print(f"  Warmup epochs: {warmup_epochs} (lambda=0)")
    print(f"  Ramp-up epochs: {finetune_epochs - warmup_epochs}")
    print(f"  Total fine-tune epochs: {finetune_epochs}")

    # Create gradual physics loss callback
    physics_callback = GradualPhysicsLossCallback(
        start_lambda=0.0,
        target_lambda=target_lambda,
        warmup_epochs=warmup_epochs,
        total_epochs=finetune_epochs
    )

    # Callbacks for fine-tuning
    callbacks = [
        physics_callback,  # Gradually increase physics loss
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=25,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=12,
            min_lr=1e-7,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            'models/pinn_finetuned.weights.h5',
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=True,
            verbose=0
        )
    ]

    # Fine-tune
    print(f"\nStarting fine-tuning with gradual physics loss...\n")

    history_phase2 = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=finetune_epochs,
        batch_size=config['batch_size'],
        callbacks=callbacks,
        verbose=1
    )

    print("\n✓ Phase 2 (Fine-tuning) complete")
    print(f"  Final val_loss: {history_phase2.history['val_loss'][-1]:.4f}")
    print(f"  Final lambda_physics: {model.lambda_physics:.6f}")

    return model, history_phase2


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
            'Model': 'PINN (Pretrained)',
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
    comp_df.to_csv('results/pinn_pretrained_vs_baselines.csv', index=False)
    print(f"\n✓ Comparison saved to: results/pinn_pretrained_vs_baselines.csv")

    return comp_df


def plot_training_history(history_phase1, history_phase2):
    """Plot combined training history from both phases"""
    print("\nGenerating training history plots...")

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # Phase 1 losses
    axes[0, 0].plot(history_phase1.history['loss'], label='Train', linewidth=2, color='blue')
    axes[0, 0].plot(history_phase1.history['val_loss'], label='Validation', linewidth=2, color='orange')
    axes[0, 0].set_xlabel('Epoch', fontsize=11)
    axes[0, 0].set_ylabel('Loss', fontsize=11)
    axes[0, 0].set_title('Phase 1: Pre-training (No Physics)', fontsize=12, fontweight='bold')
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)

    # Phase 2 losses
    axes[0, 1].plot(history_phase2.history['loss'], label='Train', linewidth=2, color='blue')
    axes[0, 1].plot(history_phase2.history['val_loss'], label='Validation', linewidth=2, color='orange')
    axes[0, 1].set_xlabel('Epoch', fontsize=11)
    axes[0, 1].set_ylabel('Loss', fontsize=11)
    axes[0, 1].set_title('Phase 2: Fine-tuning (Gradual Physics)', fontsize=12, fontweight='bold')
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)

    # Phase 2 data loss
    if 'data_loss' in history_phase2.history:
        axes[1, 0].plot(history_phase2.history['data_loss'], label='Train Data Loss', linewidth=2)
        axes[1, 0].plot(history_phase2.history['val_data_loss'], label='Val Data Loss', linewidth=2)
        axes[1, 0].set_xlabel('Epoch', fontsize=11)
        axes[1, 0].set_ylabel('Data Loss (MSE)', fontsize=11)
        axes[1, 0].set_title('Phase 2: Data Loss Evolution', fontsize=12, fontweight='bold')
        axes[1, 0].legend(fontsize=10)
        axes[1, 0].grid(True, alpha=0.3)

    # Phase 2 physics loss
    if 'physics_loss' in history_phase2.history:
        axes[1, 1].plot(history_phase2.history['physics_loss'], label='Train Physics Loss',
                       linewidth=2, color='coral')
        axes[1, 1].plot(history_phase2.history['val_physics_loss'], label='Val Physics Loss',
                       linewidth=2, color='orange')
        axes[1, 1].set_xlabel('Epoch', fontsize=11)
        axes[1, 1].set_ylabel('Physics Loss', fontsize=11)
        axes[1, 1].set_title('Phase 2: Physics Loss Evolution', fontsize=12, fontweight='bold')
        axes[1, 1].legend(fontsize=10)
        axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle('PINN Two-Phase Training History', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()

    output_dir = Path('plots/sprint3')
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.savefig(output_dir / 'pinn_pretrain_history.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: plots/sprint3/pinn_pretrain_history.png")
    plt.close()


def main():
    """Execute two-phase PINN training"""

    print("\n" + "="*80)
    print(" SPRINT 3: PINN WITH PRE-TRAINING + GRADUAL PHYSICS LOSS")
    print(" Two-Phase Curriculum Learning Strategy")
    print("="*80)

    # Load data
    data_dict = load_processed_data()

    # Configuration
    config = {
        'hidden_layers': [128, 128, 64, 32],
        'dropout': 0.2,
        'batch_size': 64,

        # Phase 1: Pre-training
        'pretrain_epochs': 50,

        # Phase 2: Fine-tuning
        'finetune_epochs': 50,
        'warmup_epochs': 10,  # Keep lambda=0 for first 10 epochs
        'target_lambda_physics': 0.001  # Target physics loss weight
    }

    print("\n" + "="*80)
    print(" TRAINING CONFIGURATION")
    print("="*80)
    print(f"\nPhase 1 (Pre-training):")
    print(f"  Epochs: {config['pretrain_epochs']}")
    print(f"  Lambda physics: 0.0 (disabled)")
    print(f"\nPhase 2 (Fine-tuning):")
    print(f"  Epochs: {config['finetune_epochs']}")
    print(f"  Warmup epochs: {config['warmup_epochs']} (lambda=0)")
    print(f"  Target lambda physics: {config['target_lambda_physics']}")
    print(f"  Strategy: Linear ramp-up after warmup")

    # Phase 1: Pre-train
    model, history_phase1 = pretrain_mlp(data_dict, config)

    # Phase 2: Fine-tune with physics
    model, history_phase2 = finetune_with_physics(model, data_dict, config, history_phase1)

    # Plot training history
    plot_training_history(history_phase1, history_phase2)

    # Evaluate on test set
    pinn_results, y_pred = evaluate_pinn(
        model,
        data_dict['X_test'],
        data_dict['y_test'],
        data_dict['target_names']
    )

    # Compare with baselines
    comparison_df = compare_with_baselines(pinn_results, data_dict['target_names'])

    # Save final model
    try:
        model.save('models/pinn_pretrained.keras')
        print(f"\n✓ Model saved: models/pinn_pretrained.keras")
    except Exception as e:
        print(f"\nWarning: Full model save failed: {e}")
        print("Saving weights only...")
        model.save_weights('models/pinn_pretrained_weights.h5')
        print(f"✓ Model weights saved: models/pinn_pretrained_weights.h5")

    # Summary
    print("\n" + "="*80)
    print(" SPRINT 3 COMPLETE - TWO-PHASE PINN TRAINING")
    print("="*80)

    print(f"""
┌─────────────────────────────────────────────────────────────────┐
│              PINN TWO-PHASE TRAINING SUMMARY                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Strategy: Pre-training + Gradual Physics Loss                 │
│                                                                 │
│  Phase 1 - Pure MLP:                                           │
│    Epochs: {config['pretrain_epochs']}, Lambda: 0.0                                       │
│                                                                 │
│  Phase 2 - Gradual Physics:                                    │
│    Epochs: {config['finetune_epochs']}, Lambda: 0.0 → {config['target_lambda_physics']:.4f}                           │
│                                                                 │
│  PINN Performance (Test Set):                                  │
""")

    for target in data_dict['target_names']:
        metrics = pinn_results[target]
        print(f"│    {target:10s}: R²={metrics['R2']:.4f}, MAE={metrics['MAE']:.4f}, MAPE={metrics['MAPE']:.2f}%   │")

    print(f"""│                                                                 │
│  Output Files:                                                  │
│    • models/pinn_pretrained.keras                               │
│    • models/pinn_pretrained.weights.h5 (phase 1)                │
│    • models/pinn_finetuned.weights.h5 (phase 2)                 │
│    • results/pinn_pretrained_vs_baselines.csv                   │
│    • plots/sprint3/pinn_pretrain_history.png                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
""")

    print("="*80)
    print(" READY FOR SPRINT 4: HYPERPARAMETER OPTIMIZATION")
    print("="*80)


if __name__ == "__main__":
    main()
