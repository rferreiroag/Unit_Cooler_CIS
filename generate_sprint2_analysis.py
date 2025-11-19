"""
Generate comprehensive analysis and visualizations for Sprint 2
"""

import sys
sys.path.append('src')

import numpy as np
import pandas as pd
import json
import joblib
from pathlib import Path

from models.advanced_models import AdvancedBaselineModel
from utils.feature_analysis import (
    plot_feature_importance,
    compare_feature_importance_across_models,
    analyze_feature_groups,
    plot_predictions_vs_actual,
    plot_residuals
)

import warnings
warnings.filterwarnings('ignore')


def main():
    """Generate all analysis and visualizations"""

    print("\n" + "="*80)
    print(" SPRINT 2: COMPREHENSIVE ANALYSIS & VISUALIZATIONS")
    print("="*80)

    # Create output directories
    plots_dir = Path('plots/sprint2')
    plots_dir.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # Load Models and Data
    # =========================================================================
    print("\n" + "="*80)
    print(" LOADING MODELS AND DATA")
    print("="*80)

    # Load models
    models = {}

    try:
        xgb_data = joblib.load('models/xgboost_model.pkl')
        xgb_model = AdvancedBaselineModel(model_type='xgboost')
        xgb_model.models_per_target = xgb_data['models']
        xgb_model.feature_names = xgb_data['feature_names']
        xgb_model.target_names = xgb_data['target_names']
        xgb_model.is_fitted = True
        models['XGBoost'] = xgb_model
        print("✓ XGBoost model loaded")
    except Exception as e:
        print(f"✗ Failed to load XGBoost: {e}")

    try:
        lgbm_data = joblib.load('models/lightgbm_model.pkl')
        lgbm_model = AdvancedBaselineModel(model_type='lightgbm')
        lgbm_model.models_per_target = lgbm_data['models']
        lgbm_model.feature_names = lgbm_data['feature_names']
        lgbm_model.target_names = lgbm_data['target_names']
        lgbm_model.is_fitted = True
        models['LightGBM'] = lgbm_model
        print("✓ LightGBM model loaded")
    except Exception as e:
        print(f"✗ Failed to load LightGBM: {e}")

    # Load test data
    X_test = np.load('data/processed/X_test_scaled.npy')
    y_test = np.load('data/processed/y_test_scaled.npy')

    with open('data/processed/metadata.json', 'r') as f:
        metadata = json.load(f)

    target_names = metadata['target_names']

    print(f"\n✓ Test data loaded: {X_test.shape}")

    # =========================================================================
    # Feature Importance Analysis
    # =========================================================================
    print("\n" + "="*80)
    print(" FEATURE IMPORTANCE ANALYSIS")
    print("="*80)

    # Individual model importance
    for model_name, model in models.items():
        print(f"\n{model_name}:")
        plot_feature_importance(
            model,
            top_n=20,
            save_path=plots_dir / f'feature_importance_{model_name.lower()}.png'
        )

    # Compare across models
    top_features = compare_feature_importance_across_models(
        models,
        top_n=15,
        save_path=plots_dir / 'feature_importance_comparison.png'
    )

    # =========================================================================
    # Feature Group Analysis
    # =========================================================================
    print("\n" + "="*80)
    print(" FEATURE GROUP ANALYSIS")
    print("="*80)

    # Define feature groups (based on Sprint 1 feature engineering)
    feature_groups = {
        'Temperature Deltas': ['delta_T_water', 'delta_T_air', 'T_approach', 'T_water_avg', 'T_air_avg'],
        'Thermal Power': ['Q_water', 'Q_air', 'Q_avg', 'Q_imbalance', 'Q_imbalance_pct'],
        'Efficiency': ['efficiency_HX', 'effectiveness', 'NTU', 'COP_estimate'],
        'Flow Derived': ['mdot_water', 'mdot_air', 'flow_ratio'],
        'Dimensionless': ['C_ratio', 'Re_air_estimate', 'delta_T_ratio'],
        'Power': ['P_fan_estimate', 'P_pump_estimate', 'P_total_estimate'],
        'Temporal': ['time_index', 'cycle_hour', 'hour_sin', 'hour_cos'],
        'Interactions': ['T_water_x_flow', 'T_air_x_flow', 'ambient_x_inlet'],
        'Raw Sensors': ['UCWIT', 'UCWOT', 'UCAIT', 'UCWF', 'AMBT', 'UCTSP']
    }

    for model_name, model in models.items():
        importance_df = model.get_feature_importance()
        if importance_df is not None:
            print(f"\n{model_name}:")
            group_df = analyze_feature_groups(importance_df, feature_groups)

    # =========================================================================
    # Predictions vs Actual
    # =========================================================================
    print("\n" + "="*80)
    print(" GENERATING PREDICTION PLOTS")
    print("="*80)

    for model_name, model in models.items():
        print(f"\n{model_name}:")
        plot_predictions_vs_actual(
            model,
            X_test,
            y_test,
            target_names,
            save_path=plots_dir / f'predictions_vs_actual_{model_name.lower()}.png'
        )

    # =========================================================================
    # Residual Analysis
    # =========================================================================
    print("\n" + "="*80)
    print(" GENERATING RESIDUAL ANALYSIS")
    print("="*80)

    for model_name, model in models.items():
        print(f"\n{model_name}:")
        plot_residuals(
            model,
            X_test,
            y_test,
            target_names,
            save_path=plots_dir / f'residuals_{model_name.lower()}.png'
        )

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "="*80)
    print(" ANALYSIS COMPLETE")
    print("="*80)

    print(f"""
┌─────────────────────────────────────────────────────────────────┐
│                   VISUALIZATIONS GENERATED                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Directory: plots/sprint2/                                      │
│                                                                 │
│  Feature Importance:                                            │
│    • feature_importance_xgboost.png                             │
│    • feature_importance_lightgbm.png                            │
│    • feature_importance_comparison.png                          │
│    • feature_importance_comparison.csv                          │
│                                                                 │
│  Model Performance:                                             │
│    • predictions_vs_actual_xgboost.png                          │
│    • predictions_vs_actual_lightgbm.png                         │
│                                                                 │
│  Residual Analysis:                                             │
│    • residuals_xgboost.png                                      │
│    • residuals_lightgbm.png                                     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
""")

    print("\n✓ All analysis and visualizations complete!")


if __name__ == "__main__":
    main()
