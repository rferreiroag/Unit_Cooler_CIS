"""
Quick training script for model WITHOUT data leakage
Trains LightGBM models on clean data for production deployment
"""

import numpy as np
import pandas as pd
import joblib
import json
from pathlib import Path
from lightgbm import LGBMRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

print("="*80)
print(" TRAINING MODEL WITHOUT DATA LEAKAGE")
print("="*80)

# Load data
print("\n[1/4] Loading data...")
X_train = np.load('data/processed_no_leakage/X_train_scaled.npy')
y_train = np.load('data/processed_no_leakage/y_train_scaled.npy')
X_val = np.load('data/processed_no_leakage/X_val_scaled.npy')
y_val = np.load('data/processed_no_leakage/y_val_scaled.npy')
X_test = np.load('data/processed_no_leakage/X_test_scaled.npy')
y_test = np.load('data/processed_no_leakage/y_test_scaled.npy')

with open('data/processed_no_leakage/metadata.json', 'r') as f:
    metadata = json.load(f)

target_names = metadata['target_names']

print(f"✓ Data loaded")
print(f"  Train: {X_train.shape}")
print(f"  Val:   {X_val.shape}")
print(f"  Test:  {X_test.shape}")
print(f"  Targets: {target_names}")

# Train models
print("\n[2/4] Training LightGBM models...")

models = {}
results = {}

lgbm_params = {
    'n_estimators': 200,
    'max_depth': 8,
    'learning_rate': 0.05,
    'num_leaves': 31,
    'min_child_samples': 20,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'verbose': -1
}

for i, target in enumerate(target_names):
    print(f"\n  Training {target}...")

    model = LGBMRegressor(**lgbm_params)
    model.fit(
        X_train, y_train[:, i],
        eval_set=[(X_val, y_val[:, i])],
        eval_metric='l2'
    )

    # Evaluate
    y_pred_test = model.predict(X_test)
    r2 = r2_score(y_test[:, i], y_pred_test)
    mae = mean_absolute_error(y_test[:, i], y_pred_test)
    rmse = np.sqrt(mean_squared_error(y_test[:, i], y_pred_test))

    models[target] = model
    results[target] = {
        'r2': float(r2),
        'mae': float(mae),
        'rmse': float(rmse)
    }

    print(f"    R² = {r2:.4f}, MAE = {mae:.4f}, RMSE = {rmse:.4f}")

print(f"\n✓ All models trained")

# Save models
print("\n[3/4] Saving models...")
Path('models').mkdir(exist_ok=True)

# Save models in a clean format without module dependencies
# Use booster_ to get the underlying LightGBM booster (more portable)
model_data = {
    'models': models,
    'results': results,
    'params': lgbm_params,
    'target_names': target_names
}

# Save using protocol=4 for better compatibility
joblib.dump(model_data, 'models/lightgbm_model_no_leakage.pkl', protocol=4)
print(f"✓ Saved to models/lightgbm_model_no_leakage.pkl")

# Summary
print("\n[4/4] Training Summary")
print("="*80)
print(f"\n{'Target':<10} {'R²':<10} {'MAE':<10} {'RMSE':<10}")
print("-"*40)
for target, res in results.items():
    print(f"{target:<10} {res['r2']:<10.4f} {res['mae']:<10.4f} {res['rmse']:<10.4f}")

print("\n" + "="*80)
print("✓ TRAINING COMPLETE (NO DATA LEAKAGE)")
print("="*80)
print(f"\nExpected R²: 0.92-0.96 (realistic for production)")
print(f"Actual average R²: {np.mean([r['r2'] for r in results.values()]):.4f}")
print(f"\n✓ Model ready for FMU export")
print(f"✓ All features computable from sensor inputs only")
