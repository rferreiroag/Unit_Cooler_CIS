"""
Validate FMU predictions against real test data

This script:
1. Loads test data with sensor inputs and real outputs
2. Simulates FMU with those sensor inputs
3. Compares FMU predictions vs real outputs
4. Calculates metrics (R², MAE, RMSE)
5. Generates validation report
"""
import numpy as np
import pandas as pd
import json
from pathlib import Path
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

print("="*80)
print(" FMU VALIDATION - Predictions vs Real Data")
print("="*80)

# Step 1: Load test data
print("\n[1/5] Loading test data...")
try:
    # Load scaled test data
    X_test_scaled = np.load('data/processed_no_leakage/X_test_scaled.npy')
    y_test_scaled = np.load('data/processed_no_leakage/y_test_scaled.npy')

    # Load clean scalers (without module dependencies)
    import joblib
    scaler_X = joblib.load('data/processed_no_leakage/scaler_clean.pkl')
    scaler_y = joblib.load('data/processed_no_leakage/y_scaler_clean.pkl')

    # Unscale X and y to get original values
    X_test = scaler_X.inverse_transform(X_test_scaled)
    y_test = scaler_y.inverse_transform(y_test_scaled)

    # Load metadata
    with open('data/processed_no_leakage/metadata.json', 'r') as f:
        metadata = json.load(f)

    feature_names = metadata['feature_names']
    target_names = metadata['target_names']

    print(f"[OK] Test data loaded")
    print(f"  Samples: {X_test.shape[0]}")
    print(f"  Features: {X_test.shape[1]}")
    print(f"  Targets: {y_test.shape[1]} ({', '.join(target_names)})")

except Exception as e:
    print(f"[FAIL] Could not load test data: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Step 2: Simulate FMU with test inputs
print("\n[2/5] Simulating FMU with test data...")

# Extract first 20 features (sensor inputs only)
sensor_names = feature_names[:20]
print(f"  Using {len(sensor_names)} sensor inputs")

try:
    from fmpy import simulate_fmu

    # Test with random subset of data (avoiding duplicates at start)
    n_samples = min(100, X_test.shape[0])
    # Take random indices to avoid duplicates
    np.random.seed(42)
    sample_indices = np.random.choice(X_test.shape[0], n_samples, replace=False)
    sample_indices = np.sort(sample_indices)  # Sort for easier debugging
    print(f"  Testing with {n_samples} random samples...")
    print(f"  Sample indices range: {sample_indices[0]} to {sample_indices[-1]}")

    fmu_predictions = []

    for idx, sample_idx in enumerate(sample_indices):
        # Prepare sensor inputs for this sample
        sensor_values = {}
        for j, sensor_name in enumerate(sensor_names):
            sensor_values[sensor_name] = float(X_test[sample_idx, j])

        # Simulate FMU for one step
        result = simulate_fmu(
            'deployment/fmu/HVACUnitCoolerFMU.fmu',
            start_values=sensor_values,
            stop_time=1.0,
            output_interval=1.0
        )

        # Extract predictions
        pred = [
            result['UCAOT'][-1],
            result['UCWOT'][-1],
            result['UCAF'][-1]
        ]
        fmu_predictions.append(pred)

        if (idx + 1) % 20 == 0:
            print(f"    Processed {idx+1}/{n_samples} samples...")

    fmu_predictions = np.array(fmu_predictions)
    y_test_subset = y_test[sample_indices]

    print(f"[OK] FMU simulation completed")

except ImportError:
    print("[FAIL] FMPy not installed. Install with: pip install fmpy")
    exit(1)
except Exception as e:
    print(f"[FAIL] Simulation error: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Step 3: Calculate metrics
print("\n[3/5] Calculating metrics...")

results = {}
for i, target in enumerate(target_names):
    y_true = y_test_subset[:, i]
    y_pred = fmu_predictions[:, i]

    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    # Calculate percentage error
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100

    results[target] = {
        'r2': r2,
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'mean_true': np.mean(y_true),
        'mean_pred': np.mean(y_pred),
        'std_true': np.std(y_true),
        'std_pred': np.std(y_pred)
    }

print("[OK] Metrics calculated")

# Step 4: Generate report
print("\n[4/5] Validation Report")
print("="*80)
print("\nPREDICTION ACCURACY:")
print("-"*80)
print(f"{'Target':<10} {'R²':<10} {'MAE':<12} {'RMSE':<12} {'MAPE %':<10}")
print("-"*80)

for target, metrics in results.items():
    print(f"{target:<10} {metrics['r2']:<10.4f} {metrics['mae']:<12.4f} "
          f"{metrics['rmse']:<12.4f} {metrics['mape']:<10.2f}")

print("\n" + "-"*80)
avg_r2 = np.mean([m['r2'] for m in results.values()])
print(f"{'Average':<10} {avg_r2:<10.4f}")
print("="*80)

print("\nSTATISTICS COMPARISON:")
print("-"*80)
print(f"{'Target':<10} {'Real Mean':<15} {'Pred Mean':<15} {'Real Std':<15} {'Pred Std':<15}")
print("-"*80)
for target, metrics in results.items():
    print(f"{target:<10} {metrics['mean_true']:<15.2f} {metrics['mean_pred']:<15.2f} "
          f"{metrics['std_true']:<15.2f} {metrics['std_pred']:<15.2f}")

# Step 5: Sample predictions
print("\n[5/5] Sample Predictions (first 5 samples):")
print("="*80)

for i in range(min(5, n_samples)):
    print(f"\nSample {i+1}:")
    for j, target in enumerate(target_names):
        real = y_test_subset[i, j]
        pred = fmu_predictions[i, j]
        error = pred - real
        error_pct = (error / real) * 100 if real != 0 else 0
        print(f"  {target}: Real={real:.2f}, Pred={pred:.2f}, "
              f"Error={error:+.2f} ({error_pct:+.1f}%)")

# Final assessment
print("\n" + "="*80)
print(" VALIDATION ASSESSMENT")
print("="*80)

if avg_r2 >= 0.75:
    print("\n[EXCELLENT] FMU predictions are highly accurate (R² >= 0.75)")
elif avg_r2 >= 0.60:
    print("\n[GOOD] FMU predictions are reasonably accurate (0.60 <= R² < 0.75)")
elif avg_r2 >= 0.40:
    print("\n[ACCEPTABLE] FMU predictions show moderate accuracy (0.40 <= R² < 0.60)")
else:
    print("\n[POOR] FMU predictions need improvement (R² < 0.40)")

print(f"\nAverage R²: {avg_r2:.4f}")
print(f"Tested on: {n_samples} samples from test set")
print("\n" + "="*80)

print("\nValidation complete!")
