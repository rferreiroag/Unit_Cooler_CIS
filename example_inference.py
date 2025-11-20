#!/usr/bin/env python3
"""
Example script showing how to use the downloaded trained models

This demonstrates both ONNX (recommended for production) and
Python pickle (recommended for retraining) inference methods.

Usage:
    python example_inference.py
"""

import numpy as np
import json

print("=" * 80)
print("HVAC Unit Cooler Digital Twin - Model Inference Example")
print("=" * 80)

# ============================================================================
# Method 1: ONNX Runtime (RECOMMENDED FOR PRODUCTION)
# ============================================================================

print("\n1. ONNX Runtime Inference (Production)")
print("-" * 80)

try:
    import onnxruntime as ort

    # Load ONNX models
    session_ucaot = ort.InferenceSession('deployment/onnx/lightgbm_ucaot.onnx')
    session_ucwot = ort.InferenceSession('deployment/onnx/lightgbm_ucwot.onnx')
    session_ucaf = ort.InferenceSession('deployment/onnx/lightgbm_ucaf.onnx')

    print("✓ ONNX models loaded successfully")

    # Create sample input (52 features) - normally you'd load real sensor data
    X_sample = np.random.randn(1, 52).astype(np.float32)

    # Make predictions
    input_name = session_ucaot.get_inputs()[0].name

    pred_ucaot = session_ucaot.run(None, {input_name: X_sample})[0][0][0]
    pred_ucwot = session_ucwot.run(None, {input_name: X_sample})[0][0][0]
    pred_ucaf = session_ucaf.run(None, {input_name: X_sample})[0][0][0]

    print(f"\nPredictions (ONNX):")
    print(f"  UCAOT (Air Outlet Temp):   {pred_ucaot:.2f} °C")
    print(f"  UCWOT (Water Outlet Temp): {pred_ucwot:.2f} °C")
    print(f"  UCAF  (Air Flow):          {pred_ucaf:.2f}")

except ImportError:
    print("✗ ONNX Runtime not installed. Install with: pip install onnxruntime")
except FileNotFoundError:
    print("✗ ONNX models not found. Extract hvac_models_package.tar.gz first")
except Exception as e:
    print(f"✗ Error: {e}")

# ============================================================================
# Method 2: Python Pickle (FOR RETRAINING AND DEVELOPMENT)
# ============================================================================

print("\n\n2. Python Pickle Inference (Development/Retraining)")
print("-" * 80)

try:
    import joblib

    # Load models and preprocessing artifacts
    models = joblib.load('models/lightgbm_model.pkl')
    scaler = joblib.load('data/processed/scaler.pkl')

    with open('data/processed/metadata.json', 'r') as f:
        metadata = json.load(f)

    print("✓ Pickle models and scaler loaded successfully")
    print(f"✓ Metadata: {metadata['n_features']} features, {metadata['n_targets']} targets")

    # Create sample input (52 features, not scaled)
    X_raw = np.random.randn(1, 52)

    # IMPORTANT: Must scale features before prediction
    X_scaled = scaler.transform(X_raw)

    # Make predictions using individual models
    model_ucaot = models['models']['UCAOT']
    model_ucwot = models['models']['UCWOT']
    model_ucaf = models['models']['UCAF']

    pred_ucaot = model_ucaot.predict(X_scaled)[0]
    pred_ucwot = model_ucwot.predict(X_scaled)[0]
    pred_ucaf = model_ucaf.predict(X_scaled)[0]

    print(f"\nPredictions (Pickle):")
    print(f"  UCAOT (Air Outlet Temp):   {pred_ucaot:.2f} °C")
    print(f"  UCWOT (Water Outlet Temp): {pred_ucwot:.2f} °C")
    print(f"  UCAF  (Air Flow):          {pred_ucaf:.2f}")

    # Show feature names (from metadata)
    print(f"\nFeature Names ({len(metadata['feature_names'])} total):")
    for i, name in enumerate(metadata['feature_names'][:10]):
        print(f"  {i+1:2d}. {name}")
    print(f"  ... ({len(metadata['feature_names']) - 10} more features)")

except ImportError:
    print("✗ Required packages not installed. Install with:")
    print("  pip install lightgbm scikit-learn joblib numpy")
except FileNotFoundError as e:
    print(f"✗ Model files not found: {e}")
    print("  Extract hvac_models_package.tar.gz first")
except Exception as e:
    print(f"✗ Error: {e}")

# ============================================================================
# Method 3: Batch Inference (Multiple Samples)
# ============================================================================

print("\n\n3. Batch Inference (Multiple Samples)")
print("-" * 80)

try:
    import onnxruntime as ort

    # Load ONNX model
    session = ort.InferenceSession('deployment/onnx/lightgbm_ucaot.onnx')

    # Create batch of 10 samples
    X_batch = np.random.randn(10, 52).astype(np.float32)

    # Make batch predictions
    input_name = session.get_inputs()[0].name
    predictions = session.run(None, {input_name: X_batch})[0]

    print(f"✓ Batch inference completed for {len(predictions)} samples")
    print(f"\nPredictions:")
    for i, pred in enumerate(predictions[:5]):
        print(f"  Sample {i+1}: UCAOT = {pred[0]:.2f} °C")
    print(f"  ... ({len(predictions) - 5} more samples)")

    # Calculate statistics
    print(f"\nBatch Statistics:")
    print(f"  Mean:   {predictions.mean():.2f} °C")
    print(f"  Std:    {predictions.std():.2f} °C")
    print(f"  Min:    {predictions.min():.2f} °C")
    print(f"  Max:    {predictions.max():.2f} °C")

except Exception as e:
    print(f"✗ Error: {e}")

# ============================================================================
# Performance Benchmarking
# ============================================================================

print("\n\n4. Performance Benchmark")
print("-" * 80)

try:
    import onnxruntime as ort
    import time

    session = ort.InferenceSession('deployment/onnx/lightgbm_ucaot.onnx')
    X = np.random.randn(1, 52).astype(np.float32)
    input_name = session.get_inputs()[0].name

    # Warmup
    for _ in range(100):
        session.run(None, {input_name: X})

    # Benchmark
    n_iterations = 1000
    start = time.time()
    for _ in range(n_iterations):
        session.run(None, {input_name: X})
    end = time.time()

    avg_latency_ms = (end - start) / n_iterations * 1000
    throughput = n_iterations / (end - start)

    print(f"✓ Benchmark completed ({n_iterations} iterations)")
    print(f"\nPerformance:")
    print(f"  Average latency:   {avg_latency_ms:.3f} ms")
    print(f"  Throughput:        {throughput:.0f} inferences/second")
    print(f"\nComparison to target (100 ms):")
    print(f"  Speedup:           {100 / avg_latency_ms:.0f}× faster")

except Exception as e:
    print(f"✗ Error: {e}")

# ============================================================================
# Summary
# ============================================================================

print("\n" + "=" * 80)
print("Summary")
print("=" * 80)
print("""
Recommended Usage:

For PRODUCTION:
  ✓ Use ONNX models (deployment/onnx/*.onnx)
  ✓ 10-100× faster than pickle
  ✓ Cross-platform, minimal dependencies
  ✓ pip install onnxruntime

For DEVELOPMENT/RETRAINING:
  ✓ Use pickle models (models/lightgbm_model.pkl)
  ✓ Full LightGBM API available
  ✓ Can retrain with new data
  ✓ pip install lightgbm scikit-learn joblib

ALWAYS REQUIRED:
  ✓ scaler.pkl - Scale features before inference
  ✓ metadata.json - Feature names and dataset info

Model Performance:
  • UCAOT: R²=0.993, MAPE=8.7%, Latency=0.022ms
  • UCWOT: R²=0.998, MAPE=8.7%, Latency=0.017ms
  • UCAF:  R²=1.000, MAPE=0.008%, Latency=0.021ms

For complete documentation, see README.md in the package.
""")
print("=" * 80)
