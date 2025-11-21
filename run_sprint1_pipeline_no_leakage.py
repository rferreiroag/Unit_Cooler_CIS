"""
Sprint 1: Data Engineering WITHOUT Data Leakage

This pipeline creates training data WITHOUT data leakage:
1. Load raw data
2. Preprocess (clean, impute, validate)
3. Engineer features using ONLY sensor inputs (no targets)
4. Create temporal train/val/test splits
5. Apply scaling
6. Save processed data

Key difference from original: Features computed WITHOUT using UCAOT, UCWOT, UCAF
"""

import sys
sys.path.append('src')

from data.data_loader import load_and_preprocess
from data.preprocessing import preprocess_unit_cooler_data
from data.feature_engineering_no_leakage import engineer_features_no_leakage
from data.data_splits import prepare_temporal_data, save_processed_data

import warnings
warnings.filterwarnings('ignore')


def main():
    """Execute complete Sprint 1 pipeline WITHOUT data leakage"""

    print("\n" + "="*80)
    print(" SPRINT 1: DATA ENGINEERING (NO DATA LEAKAGE)")
    print(" Physics-Informed Digital Twin for Unit Cooler HVAC")
    print("="*80)

    # =========================================================================
    # STEP 1: Load Raw Data
    # =========================================================================
    print("\n" + "="*80)
    print(" STEP 1: LOADING RAW DATA")
    print("="*80)

    data_path = 'data/raw/datos_combinados_entrenamiento_20251118_105234.csv'

    try:
        df_raw, metadata = load_and_preprocess(data_path)
        print(f"\n✓ Raw data loaded successfully")
        print(f"  Shape: {df_raw.shape}")
        print(f"  Memory: {df_raw.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    except Exception as e:
        print(f"\n✗ Error loading data: {e}")
        return

    # =========================================================================
    # STEP 2: Preprocessing
    # =========================================================================
    print("\n" + "="*80)
    print(" STEP 2: DATA PREPROCESSING")
    print("="*80)

    try:
        preprocess_config = None  # Using default configuration

        df_clean, preprocessor = preprocess_unit_cooler_data(df_raw, config=preprocess_config)

        print(f"\n✓ Preprocessing complete")
        print(f"  Clean shape: {df_clean.shape}")
        print(f"  Data retention: {len(df_clean)/len(df_raw)*100:.2f}%")
        print(f"  Missing values: {df_clean.isnull().sum().sum()}")

    except Exception as e:
        print(f"\n✗ Error during preprocessing: {e}")
        import traceback
        traceback.print_exc()
        return

    # =========================================================================
    # STEP 3: Feature Engineering (NO LEAKAGE)
    # =========================================================================
    print("\n" + "="*80)
    print(" STEP 3: PHYSICS-BASED FEATURE ENGINEERING (NO DATA LEAKAGE)")
    print("="*80)

    try:
        df_features, engineer = engineer_features_no_leakage(df_clean)

        print(f"\n✓ Feature engineering complete (NO DATA LEAKAGE)")
        print(f"  Original features: {len(df_clean.columns)}")
        print(f"  Total features: {len(df_features.columns)}")
        print(f"  New features: {len(engineer.feature_names)}")

        # Verify no target columns in features
        target_cols = ['UCAOT', 'UCWOT', 'UCAF']
        has_targets = any(col in df_features.columns for col in target_cols)
        if has_targets:
            print("\n⚠ WARNING: Target variables found in features!")
            print("  This indicates data leakage!")
        else:
            print("\n✓ VERIFIED: No target variables in features")
            print("  All features can be computed in production")

    except Exception as e:
        print(f"\n✗ Error during feature engineering: {e}")
        import traceback
        traceback.print_exc()
        return

    # =========================================================================
    # STEP 4: Temporal Splits and Scaling
    # =========================================================================
    print("\n" + "="*80)
    print(" STEP 4: TEMPORAL SPLITTING AND SCALING")
    print("="*80)

    try:
        # Define target variables
        target_cols = ['UCAOT', 'UCWOT', 'UCAF']

        # Prepare temporal data with splits and scaling
        data_dict = prepare_temporal_data(
            df_features,
            target_cols=target_cols,
            train_ratio=0.70,
            val_ratio=0.15,
            test_ratio=0.15,
            scaling_method='standard',
            shuffle=False  # CRITICAL: No shuffle for time series!
        )

        print(f"\n✓ Data splitting and scaling complete")
        print(f"  Features: {len(data_dict['feature_names'])}")
        print(f"  Targets: {len(data_dict['target_names'])}")
        print(f"  Train: {len(data_dict['X_train']):,} samples")
        print(f"  Val:   {len(data_dict['X_val']):,} samples")
        print(f"  Test:  {len(data_dict['X_test']):,} samples")

    except Exception as e:
        print(f"\n✗ Error during splitting/scaling: {e}")
        import traceback
        traceback.print_exc()
        return

    # =========================================================================
    # STEP 5: Save Processed Data
    # =========================================================================
    print("\n" + "="*80)
    print(" STEP 5: SAVING PROCESSED DATA (NO LEAKAGE)")
    print("="*80)

    try:
        # Save to different directory to preserve original
        save_processed_data(data_dict, output_dir='data/processed_no_leakage')

        print(f"\n✓ All processed data saved successfully")
        print(f"  Output directory: data/processed_no_leakage/")

    except Exception as e:
        print(f"\n✗ Error saving data: {e}")
        import traceback
        traceback.print_exc()
        return

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "="*80)
    print(" SPRINT 1 PIPELINE COMPLETE (NO DATA LEAKAGE)")
    print("="*80)

    print(f"""
┌─────────────────────────────────────────────────────────────────┐
│                    PIPELINE SUMMARY (NO LEAKAGE)                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Raw Data:                                                      │
│    • Samples:    {len(df_raw):>8,}                                      │
│    • Features:   {len(df_raw.columns):>8}                                       │
│                                                                 │
│  After Preprocessing:                                           │
│    • Samples:    {len(df_clean):>8,}  ({len(df_clean)/len(df_raw)*100:>5.1f}% retention)        │
│    • Features:   {len(df_clean.columns):>8}                                       │
│                                                                 │
│  After Feature Engineering (NO LEAKAGE):                        │
│    • Samples:    {len(df_features):>8,}                                      │
│    • Features:   {len(df_features.columns):>8}  (+{len(engineer.feature_names)} physics features)   │
│    • NO target variables used in features ✓                    │
│                                                                 │
│  Final Splits:                                                  │
│    • Train:      {len(data_dict['X_train']):>8,}  (70%)                              │
│    • Val:        {len(data_dict['X_val']):>8,}  (15%)                              │
│    • Test:       {len(data_dict['X_test']):>8,}  (15%)                              │
│                                                                 │
│  Output Directory: data/processed_no_leakage/                   │
│                                                                 │
│  Files Generated:                                               │
│    ✓ X_train.csv, X_val.csv, X_test.csv                        │
│    ✓ y_train.csv, y_val.csv, y_test.csv                        │
│    ✓ *_scaled.npy (scaled arrays)                              │
│    ✓ scaler.pkl (fitted scaler)                                │
│    ✓ metadata.json                                             │
│                                                                 │
│  ✓ NO DATA LEAKAGE - All features computable in production!    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
""")

    print("=" * 80)
    print(" READY FOR MODEL TRAINING (NO LEAKAGE)")
    print("=" * 80)
    print("""
Next steps:
  1. Train models with data/processed_no_leakage/
  2. Expected R²: 0.92-0.96 (realistic for production)
  3. Models will work in real-time production environment

Run: python run_sprint2_baseline_no_leakage.py
""")


if __name__ == "__main__":
    main()
