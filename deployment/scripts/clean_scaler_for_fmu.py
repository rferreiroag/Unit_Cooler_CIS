"""
Clean scaler pickle to remove module dependencies
"""
import joblib
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler

print("="*80)
print(" CLEANING SCALER FOR FMU")
print("="*80)

# Load the scaler
print("\n[1/3] Loading scaler...")
scaler = joblib.load('data/processed_no_leakage/scaler.pkl')

print(f"✓ Scaler loaded: {type(scaler)}")
print(f"  Type: {type(scaler)}")
print(f"  Attributes: {dir(scaler)}")

# Extract the underlying sklearn scaler
print("\n[2/3] Extracting sklearn scaler...")
# AdaptiveScaler has scaler_X and scaler_y attributes
sklearn_scaler = scaler.scaler_X  # Get the X scaler (features)

print(f"✓ Extracted StandardScaler")
print(f"  Mean shape: {sklearn_scaler.mean_.shape}")
print(f"  Scale shape: {sklearn_scaler.scale_.shape}")

# Create a fresh scaler with the same parameters
print("\n[3/3] Creating clean scaler...")
clean_scaler = StandardScaler()
clean_scaler.mean_ = sklearn_scaler.mean_.copy()
clean_scaler.scale_ = sklearn_scaler.scale_.copy()
clean_scaler.var_ = sklearn_scaler.var_.copy()
clean_scaler.n_features_in_ = sklearn_scaler.n_features_in_
clean_scaler.n_samples_seen_ = sklearn_scaler.n_samples_seen_

# Save with lowest protocol
print("\n[3/3] Saving clean scaler...")
output_path = 'data/processed_no_leakage/scaler_clean.pkl'
joblib.dump(clean_scaler, output_path, protocol=2)

file_size = Path(output_path).stat().st_size / 1024
print(f"✓ Saved to {output_path} ({file_size:.2f} KB)")

# Test
print("\n[Testing] Testing clean scaler...")
test_scaler = joblib.load(output_path)
test_data = np.random.randn(1, clean_scaler.n_features_in_)
result = test_scaler.transform(test_data)
print(f"✓ Clean scaler works correctly")
print(f"  Input shape: {test_data.shape}")
print(f"  Output shape: {result.shape}")

print("\n" + "="*80)
print("✓ SCALER CLEANED FOR FMU")
print("="*80)
