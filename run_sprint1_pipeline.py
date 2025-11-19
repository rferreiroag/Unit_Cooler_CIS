"""
Sprint 1: Complete Data Engineering & Feature Engineering Pipeline

This script executes the complete Sprint 1 pipeline:
1. Load raw data
2. Preprocess (clean, impute, validate)
3. Engineer physics-based features
4. Create temporal train/val/test splits
5. Apply adaptive scaling
6. Save processed data

Usage:
    python run_sprint1_pipeline.py
"""

import sys
sys.path.append('src')

from data.data_loader import load_and_preprocess
from data.preprocessing import preprocess_unit_cooler_data
from data.feature_engineering import engineer_features
from data.data_splits import prepare_temporal_data, save_processed_data

import warnings
warnings.filterwarnings('ignore')


def main():
    """Execute complete Sprint 1 pipeline"""

    print("\n" + "="*80)
    print(" SPRINT 1: DATA ENGINEERING & FEATURE ENGINEERING")
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
        # Configure preprocessing (None = use defaults)
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
    # STEP 3: Feature Engineering
    # =========================================================================
    print("\n" + "="*80)
    print(" STEP 3: PHYSICS-BASED FEATURE ENGINEERING")
    print("="*80)

    try:
        df_features, engineer = engineer_features(df_clean)

        print(f"\n✓ Feature engineering complete")
        print(f"  Original features: {len(df_clean.columns)}")
        print(f"  Total features: {len(df_features.columns)}")
        print(f"  New features: {len(engineer.feature_names)}")

        # Display feature groups
        feature_groups = engineer.get_feature_importance_groups()
        print(f"\n  Feature groups:")
        for group_name, features in feature_groups.items():
            if len(features) > 0:
                print(f"    {group_name:20s}: {len(features):2d} features")

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
    print(" STEP 5: SAVING PROCESSED DATA")
    print("="*80)

    try:
        save_processed_data(data_dict, output_dir='data/processed')

        print(f"\n✓ All processed data saved successfully")

    except Exception as e:
        print(f"\n✗ Error saving data: {e}")
        import traceback
        traceback.print_exc()
        return

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "="*80)
    print(" SPRINT 1 PIPELINE COMPLETE")
    print("="*80)

    print(f"""
┌─────────────────────────────────────────────────────────────────┐
│                         PIPELINE SUMMARY                        │
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
│  After Feature Engineering:                                     │
│    • Samples:    {len(df_features):>8,}                                      │
│    • Features:   {len(df_features.columns):>8}  (+{len(engineer.feature_names)} physics features)   │
│                                                                 │
│  Final Splits:                                                  │
│    • Train:      {len(data_dict['X_train']):>8,}  (70%)                              │
│    • Val:        {len(data_dict['X_val']):>8,}  (15%)                              │
│    • Test:       {len(data_dict['X_test']):>8,}  (15%)                              │
│                                                                 │
│  Output Directory: data/processed/                              │
│                                                                 │
│  Files Generated:                                               │
│    ✓ X_train.csv, X_val.csv, X_test.csv                        │
│    ✓ y_train.csv, y_val.csv, y_test.csv                        │
│    ✓ *_scaled.npy (scaled arrays)                              │
│    ✓ scaler.pkl (fitted scaler)                                │
│    ✓ metadata.json                                             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
""")

    print("=" * 80)
    print(" READY FOR SPRINT 2: BASELINE AVANZADO")
    print("=" * 80)
    print("""
Next steps:
  1. Train advanced baseline models (XGBoost, LightGBM, MLP)
  2. Feature importance analysis
  3. Cross-validation temporal
  4. Benchmark for PINN comparison

Run: python run_sprint2_baseline.py
""")


if __name__ == "__main__":
    main()
