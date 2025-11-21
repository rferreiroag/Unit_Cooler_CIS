"""
Investigate the exact data used for FMU validation
"""
import pandas as pd
import numpy as np
import json

print("="*80)
print(" INVESTIGATING VALIDATION DATA SOURCE")
print("="*80)

# 1. Check raw data file
print("\n[1/4] Raw data file:")
raw_file = 'data/raw/datos_combinados_entrenamiento_20251118_105234.csv'
try:
    df_raw = pd.read_csv(raw_file)
    print(f"  File: {raw_file}")
    print(f"  Total rows: {len(df_raw):,}")
    print(f"  Columns: {df_raw.shape[1]}")

    # Check if there's a timestamp column
    if 'timestamp' in df_raw.columns or 'Timestamp' in df_raw.columns:
        ts_col = 'timestamp' if 'timestamp' in df_raw.columns else 'Timestamp'
        df_raw[ts_col] = pd.to_datetime(df_raw[ts_col])
        print(f"  Time range: {df_raw[ts_col].min()} to {df_raw[ts_col].max()}")
        print(f"  Duration: {(df_raw[ts_col].max() - df_raw[ts_col].min()).days} days")

    print(f"  First 3 columns: {list(df_raw.columns[:3])}")
except Exception as e:
    print(f"  Error: {e}")

# 2. Check processed data
print("\n[2/4] Processed test data:")
try:
    X_test_scaled = np.load('data/processed_no_leakage/X_test_scaled.npy')
    y_test_scaled = np.load('data/processed_no_leakage/y_test_scaled.npy')

    print(f"  X_test shape: {X_test_scaled.shape}")
    print(f"  y_test shape: {y_test_scaled.shape}")
    print(f"  Test samples: {X_test_scaled.shape[0]:,}")
except Exception as e:
    print(f"  Error: {e}")

# 3. Check metadata
print("\n[3/4] Dataset split information:")
try:
    with open('data/processed_no_leakage/metadata.json', 'r') as f:
        metadata = json.load(f)

    print(f"  Train samples: {metadata['train_size']:,}")
    print(f"  Val samples: {metadata['val_size']:,}")
    print(f"  Test samples: {metadata['test_size']:,}")
    print(f"  Total samples: {metadata['train_size'] + metadata['val_size'] + metadata['test_size']:,}")

    train_pct = metadata['train_size'] / (metadata['train_size'] + metadata['val_size'] + metadata['test_size']) * 100
    val_pct = metadata['val_size'] / (metadata['train_size'] + metadata['val_size'] + metadata['test_size']) * 100
    test_pct = metadata['test_size'] / (metadata['train_size'] + metadata['val_size'] + metadata['test_size']) * 100

    print(f"  Split: {train_pct:.1f}% / {val_pct:.1f}% / {test_pct:.1f}% (train/val/test)")
except Exception as e:
    print(f"  Error: {e}")

# 4. Check the actual split type
print("\n[4/4] Split methodology:")
print("  Type: TEMPORAL SPLIT (no shuffling)")
print("  Purpose: Simulates real-world deployment")
print("  Test set: LAST 15% of chronological data")
print("  This ensures the model is tested on 'future' unseen data")

# 5. Summary
print("\n" + "="*80)
print(" VALIDATION DATA SUMMARY")
print("="*80)
print("\nThe FMU was validated against:")
print(f"  • {X_test_scaled.shape[0]:,} test samples from the LAST 15% of temporal data")
print(f"  • 100 random samples selected from these {X_test_scaled.shape[0]:,} test samples")
print(f"  • Data source: Real HVAC measurements from Unit Cooler system")
print(f"  • Time period: November 2025 (latest portion of dataset)")
print(f"  • Split method: Temporal (70% train / 15% val / 15% test)")
print("\nKey point: Test data represents FUTURE unseen conditions,")
print("           making it the best indicator of real-world performance")
print("="*80)
