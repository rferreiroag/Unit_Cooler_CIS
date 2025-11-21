"""
Export LightGBM models to ONNX format for portable FMU deployment

ONNX allows the model to run without Python/LightGBM dependencies
"""
import numpy as np
import joblib
import json
from pathlib import Path

print("="*80)
print(" EXPORT MODELS TO ONNX")
print("="*80)

# Check if required packages are available
try:
    import onnx
    import onnxruntime as ort
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    print("✓ ONNX packages available")
except ImportError as e:
    print(f"✗ Missing ONNX packages: {e}")
    print("\nInstall with:")
    print("  pip install onnx onnxruntime skl2onnx onnxmltools")
    exit(1)

# Load clean model
print("\n[1/5] Loading clean model...")
model_data = joblib.load('models/lightgbm_model_no_leakage_clean.pkl')
models = model_data['models']
target_names = model_data['target_names']

print(f"✓ Loaded {len(models)} models")
print(f"  Targets: {target_names}")

# Load scaler
print("\n[2/5] Loading clean scaler...")
scaler = joblib.load('data/processed_no_leakage/scaler_clean.pkl')
print(f"✓ Loaded scaler")
print(f"  Input features: {scaler.n_features_in_}")

# Export each model to ONNX
print("\n[3/5] Exporting models to ONNX...")
output_dir = Path('models/onnx')
output_dir.mkdir(exist_ok=True)

n_features = scaler.n_features_in_

for target_name, model in models.items():
    print(f"\n  Exporting {target_name}...")

    # Define input type
    initial_type = [('float_input', FloatTensorType([None, n_features]))]

    # Convert to ONNX
    try:
        onnx_model = convert_sklearn(
            model,
            initial_types=initial_type,
            target_opset=12
        )

        # Save ONNX model
        output_path = output_dir / f"{target_name}.onnx"
        with open(output_path, "wb") as f:
            f.write(onnx_model.SerializeToString())

        file_size = output_path.stat().st_size / 1024
        print(f"    ✓ Saved to {output_path} ({file_size:.2f} KB)")

    except Exception as e:
        print(f"    ✗ Failed: {e}")
        continue

# Export scaler to ONNX
print("\n[4/5] Exporting scaler to ONNX...")
initial_type = [('float_input', FloatTensorType([None, n_features]))]
try:
    onnx_scaler = convert_sklearn(
        scaler,
        initial_types=initial_type,
        target_opset=12
    )

    scaler_path = output_dir / "scaler.onnx"
    with open(scaler_path, "wb") as f:
        f.write(onnx_scaler.SerializeToString())

    file_size = scaler_path.stat().st_size / 1024
    print(f"  ✓ Saved to {scaler_path} ({file_size:.2f} KB)")
except Exception as e:
    print(f"  ✗ Failed: {e}")

# Test ONNX models
print("\n[5/5] Testing ONNX models...")
test_input = np.random.randn(1, n_features).astype(np.float32)

all_work = True
for target_name in target_names:
    onnx_path = output_dir / f"{target_name}.onnx"
    if onnx_path.exists():
        try:
            sess = ort.InferenceSession(str(onnx_path))
            input_name = sess.get_inputs()[0].name
            result = sess.run(None, {input_name: test_input})
            print(f"  ✓ {target_name}: {result[0][0][0]:.4f}")
        except Exception as e:
            print(f"  ✗ {target_name}: {e}")
            all_work = False

# Save metadata
metadata_onnx = {
    'target_names': target_names,
    'n_features': n_features,
    'format': 'ONNX',
    'opset': 12
}

with open(output_dir / 'metadata.json', 'w') as f:
    json.dump(metadata_onnx, f, indent=2)

print("\n" + "="*80)
if all_work:
    print("✓ ALL MODELS EXPORTED TO ONNX")
    print("="*80)
    print(f"\nONNX models location: {output_dir}/")
    print(f"\nAdvantages:")
    print(f"  • No Python runtime required")
    print(f"  • No LightGBM/sklearn dependencies")
    print(f"  • Faster inference")
    print(f"  • Cross-platform compatible")
    print(f"\nNext: Create ONNX-based FMU")
else:
    print("⚠ SOME EXPORTS FAILED")
    print("="*80)
