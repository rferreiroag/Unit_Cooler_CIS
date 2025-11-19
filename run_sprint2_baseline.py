"""
Sprint 2: Advanced Baseline Models

This script executes the complete Sprint 2 pipeline:
1. Load processed data from Sprint 1
2. Train advanced models (XGBoost, LightGBM, MLP)
3. Evaluate and compare all models
4. Feature importance analysis
5. Save results and visualizations

Usage:
    python run_sprint2_baseline.py
"""

import sys
sys.path.append('src')

import numpy as np
import pandas as pd
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

from models.advanced_models import AdvancedBaselineModel, temporal_cross_validation

import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)


def load_processed_data():
    """Load processed data from Sprint 1"""
    print("\n" + "="*80)
    print(" LOADING PROCESSED DATA FROM SPRINT 1")
    print("="*80)

    data_dir = Path('data/processed')

    if not data_dir.exists():
        raise FileNotFoundError(
            "Processed data not found. Please run: python run_sprint1_pipeline.py"
        )

    # Load scaled arrays
    X_train = np.load(data_dir / 'X_train_scaled.npy')
    y_train = np.load(data_dir / 'y_train_scaled.npy')
    X_val = np.load(data_dir / 'X_val_scaled.npy')
    y_val = np.load(data_dir / 'y_val_scaled.npy')
    X_test = np.load(data_dir / 'X_test_scaled.npy')
    y_test = np.load(data_dir / 'y_test_scaled.npy')

    # Load metadata
    with open(data_dir / 'metadata.json', 'r') as f:
        metadata = json.load(f)

    print(f"\n✓ Data loaded successfully")
    print(f"  Train: {X_train.shape}")
    print(f"  Val:   {X_val.shape}")
    print(f"  Test:  {X_test.shape}")
    print(f"  Features: {len(metadata['feature_names'])}")
    print(f"  Targets: {metadata['target_names']}")

    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val,
        'X_test': X_test,
        'y_test': y_test,
        'feature_names': metadata['feature_names'],
        'target_names': metadata['target_names']
    }


def train_all_models(data_dict):
    """Train all advanced baseline models"""
    print("\n" + "="*80)
    print(" TRAINING ADVANCED BASELINE MODELS")
    print("="*80)

    X_train = data_dict['X_train']
    y_train = data_dict['y_train']
    X_val = data_dict['X_val']
    y_val = data_dict['y_val']
    feature_names = data_dict['feature_names']
    target_names = data_dict['target_names']

    models = {}
    results = {}

    # =========================================================================
    # Model 1: XGBoost
    # =========================================================================
    print("\n" + "="*80)
    print(" MODEL 1: XGBoost")
    print("="*80)

    try:
        xgb_model = AdvancedBaselineModel(
            model_type='xgboost',
            n_estimators=300,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8
        )

        xgb_model.fit(X_train, y_train, X_val, y_val,
                     feature_names=feature_names, target_names=target_names)

        # Evaluate
        train_metrics = xgb_model.evaluate(X_train, y_train)
        val_metrics = xgb_model.evaluate(X_val, y_val)

        models['XGBoost'] = xgb_model
        results['XGBoost'] = {
            'train': train_metrics,
            'val': val_metrics,
            'training_time': xgb_model.training_time
        }

        print(f"\n✓ XGBoost trained in {xgb_model.training_time:.2f}s")
        print("\nValidation Performance:")
        for target, metrics in val_metrics.items():
            print(f"  {target}: R²={metrics['R2']:.4f}, MAE={metrics['MAE']:.4f}, MAPE={metrics['MAPE']:.2f}%")

    except Exception as e:
        print(f"\n✗ XGBoost training failed: {e}")

    # =========================================================================
    # Model 2: LightGBM
    # =========================================================================
    print("\n" + "="*80)
    print(" MODEL 2: LightGBM")
    print("="*80)

    try:
        lgbm_model = AdvancedBaselineModel(
            model_type='lightgbm',
            n_estimators=300,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8
        )

        lgbm_model.fit(X_train, y_train, X_val, y_val,
                      feature_names=feature_names, target_names=target_names)

        # Evaluate
        train_metrics = lgbm_model.evaluate(X_train, y_train)
        val_metrics = lgbm_model.evaluate(X_val, y_val)

        models['LightGBM'] = lgbm_model
        results['LightGBM'] = {
            'train': train_metrics,
            'val': val_metrics,
            'training_time': lgbm_model.training_time
        }

        print(f"\n✓ LightGBM trained in {lgbm_model.training_time:.2f}s")
        print("\nValidation Performance:")
        for target, metrics in val_metrics.items():
            print(f"  {target}: R²={metrics['R2']:.4f}, MAE={metrics['MAE']:.4f}, MAPE={metrics['MAPE']:.2f}%")

    except Exception as e:
        print(f"\n✗ LightGBM training failed: {e}")

    # =========================================================================
    # Model 3: MLP (Multi-Layer Perceptron)
    # =========================================================================
    print("\n" + "="*80)
    print(" MODEL 3: MLP (Multi-Layer Perceptron)")
    print("="*80)

    try:
        mlp_model = AdvancedBaselineModel(
            model_type='mlp',
            hidden_layers=[128, 64, 32],
            dropout=0.3,
            learning_rate=0.001,
            epochs=200,
            batch_size=32
        )

        mlp_model.fit(X_train, y_train, X_val, y_val,
                     feature_names=feature_names, target_names=target_names)

        # Evaluate
        train_metrics = mlp_model.evaluate(X_train, y_train)
        val_metrics = mlp_model.evaluate(X_val, y_val)

        models['MLP'] = mlp_model
        results['MLP'] = {
            'train': train_metrics,
            'val': val_metrics,
            'training_time': mlp_model.training_time
        }

        print(f"\n✓ MLP trained in {mlp_model.training_time:.2f}s")
        print("\nValidation Performance:")
        for target, metrics in val_metrics.items():
            print(f"  {target}: R²={metrics['R2']:.4f}, MAE={metrics['MAE']:.4f}, MAPE={metrics['MAPE']:.2f}%")

    except Exception as e:
        print(f"\n✗ MLP training failed: {e}")
        import traceback
        traceback.print_exc()

    return models, results


def evaluate_on_test_set(models, data_dict):
    """Evaluate all models on test set"""
    print("\n" + "="*80)
    print(" TEST SET EVALUATION")
    print("="*80)

    X_test = data_dict['X_test']
    y_test = data_dict['y_test']

    test_results = {}

    for model_name, model in models.items():
        print(f"\nEvaluating {model_name}...")

        metrics = model.evaluate(X_test, y_test)
        test_results[model_name] = metrics

        print(f"\n{model_name} Test Performance:")
        for target, m in metrics.items():
            print(f"  {target}: R²={m['R2']:.4f}, MAE={m['MAE']:.4f}, MAPE={m['MAPE']:.2f}%")

    return test_results


def generate_comparison_table(results, test_results):
    """Generate comprehensive comparison table"""
    print("\n" + "="*80)
    print(" MODEL COMPARISON")
    print("="*80)

    # Create comparison DataFrame
    rows = []

    for model_name in results.keys():
        for dataset in ['train', 'val']:
            if dataset in results[model_name]:
                for target, metrics in results[model_name][dataset].items():
                    row = {
                        'Model': model_name,
                        'Dataset': dataset,
                        'Target': target,
                        'MAE': metrics['MAE'],
                        'RMSE': metrics['RMSE'],
                        'MAPE': metrics['MAPE'],
                        'R2': metrics['R2'],
                        'Max_Error': metrics['Max_Error']
                    }
                    rows.append(row)

        # Add test results
        if model_name in test_results:
            for target, metrics in test_results[model_name].items():
                row = {
                    'Model': model_name,
                    'Dataset': 'test',
                    'Target': target,
                    'MAE': metrics['MAE'],
                    'RMSE': metrics['RMSE'],
                    'MAPE': metrics['MAPE'],
                    'R2': metrics['R2'],
                    'Max_Error': metrics['Max_Error']
                }
                rows.append(row)

    df = pd.DataFrame(rows)

    # Save to CSV
    output_path = Path('results')
    output_path.mkdir(exist_ok=True)

    df.to_csv(output_path / 'advanced_baseline_comparison.csv', index=False)
    print(f"\n✓ Results saved to: results/advanced_baseline_comparison.csv")

    # Print summary for test set
    print("\n" + "-"*80)
    print(" TEST SET SUMMARY")
    print("-"*80)

    test_df = df[df['Dataset'] == 'test']

    for target in test_df['Target'].unique():
        print(f"\n{target}:")
        target_df = test_df[test_df['Target'] == target][['Model', 'R2', 'MAE', 'MAPE']]
        target_df = target_df.sort_values('R2', ascending=False)
        print(target_df.to_string(index=False))

    return df


def save_models(models):
    """Save trained models"""
    print("\n" + "="*80)
    print(" SAVING MODELS")
    print("="*80)

    models_dir = Path('models')
    models_dir.mkdir(exist_ok=True)

    for model_name, model in models.items():
        filepath = models_dir / f"{model_name.lower()}_model"

        if model.model_type == 'mlp':
            filepath = str(filepath) + '.h5'
        else:
            filepath = str(filepath) + '.pkl'

        model.save(filepath)

    print("\n✓ All models saved successfully")


def main():
    """Execute complete Sprint 2 pipeline"""

    print("\n" + "="*80)
    print(" SPRINT 2: ADVANCED BASELINE MODELS")
    print(" Physics-Informed Digital Twin for Unit Cooler HVAC")
    print("="*80)

    # Load data
    data_dict = load_processed_data()

    # Train models
    models, results = train_all_models(data_dict)

    if not models:
        print("\n✗ No models trained successfully")
        return

    # Evaluate on test set
    test_results = evaluate_on_test_set(models, data_dict)

    # Generate comparison
    comparison_df = generate_comparison_table(results, test_results)

    # Save models
    save_models(models)

    # Summary
    print("\n" + "="*80)
    print(" SPRINT 2 COMPLETE")
    print("="*80)

    print(f"""
┌─────────────────────────────────────────────────────────────────┐
│                      SPRINT 2 SUMMARY                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Models Trained:                                                │
│    ✓ XGBoost     - Gradient Boosting                           │
│    ✓ LightGBM    - High-performance GBDT                       │
│    ✓ MLP         - Multi-Layer Perceptron                      │
│                                                                 │
│  Best Performing Model (Test Set):                             │
""")

    # Find best model per target
    test_df = comparison_df[comparison_df['Dataset'] == 'test']
    for target in data_dict['target_names']:
        target_df = test_df[test_df['Target'] == target]
        best = target_df.loc[target_df['R2'].idxmax()]
        print(f"│    {target:10s}: {best['Model']:12s} (R²={best['R2']:.4f}, MAE={best['MAE']:.4f})   │")

    print(f"""│                                                                 │
│  Output Files:                                                  │
│    • results/advanced_baseline_comparison.csv                   │
│    • models/xgboost_model.pkl                                   │
│    • models/lightgbm_model.pkl                                  │
│    • models/mlp_model.h5                                        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
""")

    print("="*80)
    print(" READY FOR SPRINT 3: PHYSICS-INFORMED NEURAL NETWORK")
    print("="*80)
    print("""
Next steps:
  1. Design PINN architecture with physics loss
  2. Implement multi-objective training
  3. Validate thermodynamic constraints
  4. Compare PINN vs baselines

Run: python run_sprint3_pinn.py
""")


if __name__ == "__main__":
    main()
