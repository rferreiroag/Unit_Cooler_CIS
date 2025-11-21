"""
Extract y_scaler from AdaptiveScaler for output descaling in FMU
"""
import joblib
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler

print("="*80)
print(" EXTRACTING Y_SCALER FOR FMU")
print("="*80)

# Load the scaler
print("\n[1/3] Loading scaler...")
scaler = joblib.load('data/processed_no_leakage/scaler.pkl')

print(f"[OK] Scaler loaded: {type(scaler)}")

# Extract the y scaler (targets)
print("\n[2/3] Extracting y_scaler (for output descaling)...")
y_scaler = scaler.scaler_y  # Get the y scaler (targets)

print(f"[OK] Extracted y_scaler")
print(f"  Mean shape: {y_scaler.mean_.shape}")
print(f"  Scale shape: {y_scaler.scale_.shape}")
print(f"  Mean values: {y_scaler.mean_}")
print(f"  Scale values: {y_scaler.scale_}")

# Create clean y_scaler
print("\n[3/3] Saving clean y_scaler...")
clean_y_scaler = StandardScaler()
clean_y_scaler.mean_ = y_scaler.mean_.copy()
clean_y_scaler.scale_ = y_scaler.scale_.copy()
clean_y_scaler.var_ = y_scaler.var_.copy()
clean_y_scaler.n_features_in_ = y_scaler.n_features_in_
clean_y_scaler.n_samples_seen_ = y_scaler.n_samples_seen_

output_path = 'data/processed_no_leakage/y_scaler_clean.pkl'
joblib.dump(clean_y_scaler, output_path, protocol=2)

file_size = Path(output_path).stat().st_size / 1024
print(f"[OK] Saved to {output_path} ({file_size:.2f} KB)")

# Test
print("\n[Testing] Testing y_scaler...")
test_scaled = np.array([[0.5, -0.35, -0.81]])  # Example scaled outputs
test_unscaled = clean_y_scaler.inverse_transform(test_scaled)
print(f"  Scaled: {test_scaled[0]}")
print(f"  Unscaled: {test_unscaled[0]}")

print("\n" + "="*80)
print("[OK] Y_SCALER READY FOR FMU")
print("="*80)
