"""
Clean model pickle to remove module dependencies
This ensures the model can be loaded in FMU without custom modules
"""
import joblib
import sys
import json
from pathlib import Path

print("="*80)
print(" CLEANING MODEL FOR FMU")
print("="*80)

# Load the trained model
print("\n[1/3] Loading model...")
model_data = joblib.load('models/lightgbm_model_no_leakage.pkl')

print(f"✓ Model loaded")
print(f"  Keys: {list(model_data.keys())}")
print(f"  Models: {list(model_data['models'].keys())}")

# Extract only what we need for FMU (without module dependencies)
print("\n[2/3] Extracting clean model data...")

clean_model_data = {
    'models': model_data['models'],  # LightGBM models
    'target_names': model_data['target_names'],  # Just the names
    'params': model_data['params']  # Training params
}

# Save with lowest protocol for maximum compatibility
print("\n[3/3] Saving clean model...")
output_path = 'models/lightgbm_model_no_leakage_clean.pkl'
joblib.dump(clean_model_data, output_path, protocol=2)

file_size = Path(output_path).stat().st_size / (1024 * 1024)
print(f"✓ Saved to {output_path} ({file_size:.2f} MB)")

# Test loading
print("\n[Testing] Reloading clean model...")
test_load = joblib.load(output_path)
print(f"✓ Clean model loads successfully")
print(f"  Models: {list(test_load['models'].keys())}")

print("\n" + "="*80)
print("✓ MODEL CLEANED FOR FMU")
print("="*80)
print(f"\nNext steps:")
print(f"1. Copy clean model to FMU resources")
print(f"2. Rebuild FMU")
print(f"3. Test FMU")
