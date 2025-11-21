"""
Debug test data to understand the issue
"""
import numpy as np
import joblib
import json

print("="*80)
print(" DEBUGGING TEST DATA")
print("="*80)

# Load scaled data
print("\n[1] Loading scaled test data...")
y_test_scaled = np.load('data/processed_no_leakage/y_test_scaled.npy')

print(f"y_test_scaled shape: {y_test_scaled.shape}")
print(f"y_test_scaled (first 5 rows):")
print(y_test_scaled[:5])
print(f"\ny_test_scaled statistics:")
print(f"  Mean: {np.mean(y_test_scaled, axis=0)}")
print(f"  Std: {np.std(y_test_scaled, axis=0)}")
print(f"  Min: {np.min(y_test_scaled, axis=0)}")
print(f"  Max: {np.max(y_test_scaled, axis=0)}")

# Load y_scaler
print("\n[2] Loading y_scaler...")
scaler_y = joblib.load('data/processed_no_leakage/y_scaler_clean.pkl')

print(f"y_scaler mean: {scaler_y.mean_}")
print(f"y_scaler scale: {scaler_y.scale_}")
print(f"y_scaler n_features: {scaler_y.n_features_in_}")

# Unscale
print("\n[3] Unscaling test data...")
y_test_unscaled = scaler_y.inverse_transform(y_test_scaled)

print(f"y_test_unscaled (first 5 rows):")
print(y_test_unscaled[:5])
print(f"\ny_test_unscaled statistics:")
print(f"  Mean: {np.mean(y_test_unscaled, axis=0)}")
print(f"  Std: {np.std(y_test_unscaled, axis=0)}")
print(f"  Min: {np.min(y_test_unscaled, axis=0)}")
print(f"  Max: {np.max(y_test_unscaled, axis=0)}")

# Load metadata
print("\n[4] Checking metadata...")
with open('data/processed_no_leakage/metadata.json', 'r') as f:
    metadata = json.load(f)

print(f"Target names: {metadata['target_names']}")

print("\n" + "="*80)
