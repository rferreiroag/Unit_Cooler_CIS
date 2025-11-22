# CLAUDE.md - AI Assistant Guide for Unit_Cooler_CIS

**Last Updated:** 2025-11-22
**Project Version:** 1.0.0
**Status:** Production Ready

This document provides comprehensive guidance for AI assistants working with the Unit_Cooler_CIS repository - a production-ready **Data-Driven Digital Twin** for HVAC Unit Cooler systems using LightGBM models with physics-based feature engineering.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Codebase Architecture](#codebase-architecture)
3. [Development Workflows](#development-workflows)
4. [Key Conventions](#key-conventions)
5. [Data Pipeline](#data-pipeline)
6. [Testing Guidelines](#testing-guidelines)
7. [Deployment Procedures](#deployment-procedures)
8. [Common Tasks](#common-tasks)
9. [Critical Constraints](#critical-constraints)
10. [Git Workflow](#git-workflow)

---

## Project Overview

### What This Project Does

This is a **machine learning digital twin** for HVAC Unit Cooler systems that:
- Predicts 3 target variables: **UCAOT** (Air Outlet Temp), **UCWOT** (Water Outlet Temp), **UCAF** (Air Flow)
- Uses 20 sensor inputs with 19 physics-based engineered features (39 total)
- Achieves R²=0.78-0.92 on real test data
- Deploys as FMI 2.0 Co-Simulation FMU for building automation systems
- Has <1ms inference time and is production-ready

### Key Achievement: Zero Data Leakage

**CRITICAL:** This project solved a data leakage problem. The original pipeline had 17 features dependent on target variables, causing artificially high R² (>0.99) that couldn't be reproduced in production. The current version has:
- ✅ **ZERO features dependent on targets** (UCAOT, UCWOT, UCAF)
- ✅ All 39 features computable from 20 sensor inputs only
- ✅ Realistic R² (0.78-0.92) validated on real data
- ✅ FMU tested and production-ready

### Technology Stack

- **Language:** Python 3.11+
- **ML Framework:** LightGBM 4.1+ (primary), XGBoost 2.0+ (comparison)
- **Data:** NumPy 1.24+, Pandas 2.0+, Scikit-learn 1.3+
- **FMU Export:** PythonFMU 0.6+, FMPy 0.3+
- **Deployment:** FastAPI, ONNX, MQTT, BACnet
- **Testing:** pytest 7.4+
- **Docs:** Sphinx 7.2+

---

## Codebase Architecture

### Directory Structure

```
Unit_Cooler_CIS/
├── src/                          # Core library modules (import from here)
│   ├── data/                      # Data pipeline
│   │   ├── data_loader.py         # CSV loading
│   │   ├── preprocessing.py       # Cleaning, imputation, outliers
│   │   ├── feature_engineering_no_leakage.py  # 19 physics features
│   │   ├── data_splits.py         # Temporal train/val/test splits
│   │   └── __init__.py
│   ├── models/                    # ML models
│   │   ├── baseline_models.py     # LightGBM, XGBoost, RF wrappers
│   │   ├── advanced_models.py     # MLP, ensemble
│   │   ├── pinn_model.py          # Physics-Informed NNs (experimental)
│   │   └── __init__.py
│   └── utils/                     # Analysis utilities
│       ├── eda_utils.py           # Exploratory data analysis
│       ├── visualization.py       # Plotting
│       ├── feature_analysis.py    # Feature importance
│       └── __init__.py
│
├── deployment/                    # Production artifacts
│   ├── fmu/                       # FMU 2.0 Co-Simulation
│   │   ├── HVACUnitCoolerFMU.fmu  # Compiled FMU (2.4 MB) ⭐
│   │   ├── hvac_fmu_sensor_inputs.py  # FMU implementation
│   │   ├── resources/             # Model + scaler artifacts
│   │   └── README*.md             # FMU documentation (4 guides)
│   ├── scripts/                   # Build & export scripts
│   │   ├── export_fmu_sensor_inputs.py     # Main FMU builder
│   │   ├── export_model_to_onnx.py         # ONNX export
│   │   ├── extract_y_scaler_for_fmu.py     # Output scaler extraction
│   │   ├── clean_model_for_fmu.py          # Model cleaning
│   │   └── clean_scaler_for_fmu.py         # Scaler cleaning
│   ├── validation/                # FMU testing
│   │   ├── validate_fmu_predictions.py     # Full validation (100 samples)
│   │   ├── test_fmu.py                     # Basic test
│   │   ├── test_fmu_comprehensive.py       # Comprehensive test
│   │   └── example_inference.py            # Usage examples
│   ├── packages/                  # Release packages
│   ├── onnx/                      # ONNX deployment
│   ├── docker/                    # Containerization
│   └── benchmarks/                # Performance tests
│
├── integration/                   # Real-time integration
│   ├── mqtt/                      # MQTT client for IoT
│   └── bacnet/                    # BACnet for building automation
│
├── api/                           # FastAPI inference endpoint
├── dashboard/                     # Streamlit dashboard
├── monitoring/                    # Drift detection
├── data/                          # Data storage
│   ├── raw/                       # Original CSV (56,211 samples)
│   └── processed_no_leakage/      # Processed arrays + scalers
│
├── models/                        # Trained model artifacts
├── results/                       # Evaluation results (CSV)
├── plots/                         # Visualizations
├── tests/                         # pytest unit tests
├── scripts/                       # Utility scripts
│   └── analysis/                  # Data analysis
├── docs/                          # Complete documentation
│   ├── Sprint*.md                 # Sprint-specific docs (0-8)
│   └── final/                     # Deliverables
│
├── run_sprint1_pipeline_no_leakage.py  # ⭐ Data prep pipeline
├── train_model_no_leakage.py           # ⭐ Model training
├── run_sprint2_baseline.py             # Baseline comparison
├── run_sprint5_evaluation.py           # Comprehensive eval
├── run_sprint6_deployment.py           # FMU deployment
├── requirements.txt                    # Main dependencies
├── README.md                           # Project documentation
├── CHANGELOG_NO_LEAKAGE.md             # Development history
└── CLAUDE.md                           # This file
```

### Module Organization

#### `src/data/` - Data Pipeline Modules

**Purpose:** Reusable data processing components

Key classes:
- `DataLoader` - Loads CSV with semicolon delimiter detection
- `DataPreprocessor` - 11-step preprocessing (saturation, outliers, imputation)
- `PhysicsFeatureEngineerNoLeakage` - Generates 19 features from 20 sensors
- `TemporalDataSplitter` - Temporal train/val/test splits (70/15/15)
- `AdaptiveScaler` - StandardScaler wrapper

#### `src/models/` - ML Model Implementations

**Purpose:** Model wrappers and training logic

Key classes:
- `BaselineModel` - Unified wrapper for Linear/RF/MultiOutput
- LightGBM/XGBoost integration via sklearn
- `AdvancedModels` - MLP, ensemble (experimental)
- `PINNModel` - Physics-Informed Neural Networks (experimental)

#### `src/utils/` - Analysis Utilities

**Purpose:** EDA, visualization, feature analysis

Key modules:
- `eda_utils.py` - Statistical summaries, correlation
- `visualization.py` - Matplotlib/Seaborn plots
- `feature_analysis.py` - Importance extraction, SHAP

---

## Development Workflows

### Main Pipeline Scripts (Execute in Order)

The project follows a **sprint-based workflow**. Execute these scripts sequentially:

#### 1. Sprint 1: Data Preparation
```bash
python run_sprint1_pipeline_no_leakage.py
```
**What it does:**
- Loads raw data (56,211 samples)
- Preprocessing: sensor saturation → negative flows → outliers → imputation
- Feature engineering: 20 sensors → 39 features (NO leakage)
- Temporal split: 70/15/15 (train/val/test)
- Scaling: StandardScaler fitted on train only
- **Output:** `data/processed_no_leakage/` (X_train/val/test, y_train/val/test, scalers, metadata)

#### 2. Model Training
```bash
python train_model_no_leakage.py
```
**What it does:**
- Loads scaled data from Sprint 1
- Trains 3 LightGBM models (UCAOT, UCWOT, UCAF)
- Evaluates on test set
- **Output:** `models/lightgbm_model_no_leakage.pkl`

**Expected performance:**
- UCAOT: R²=0.91, MAE=0.136
- UCWOT: R²=0.75, MAE=0.253
- UCAF: R²=0.75, MAE=0.200

#### 3. Sprint 2: Baseline Comparison (Optional)
```bash
python run_sprint2_baseline.py
```
**What it does:**
- Compares LightGBM vs Linear/RandomForest
- **Output:** `results/baseline_comparison.csv`

#### 4. Sprint 5: Comprehensive Evaluation (Optional)
```bash
python run_sprint5_evaluation.py
```
**What it does:**
- Feature importance analysis
- Residual analysis
- Cross-validation (5 temporal folds)
- Performance by operating conditions
- **Output:** Multiple CSV files in `results/`

#### 5. Sprint 6: FMU Deployment
```bash
python run_sprint6_deployment.py
# OR directly:
python deployment/scripts/export_fmu_sensor_inputs.py
```
**What it does:**
- Builds FMU from trained model
- Embeds model, scalers, metadata
- **Output:** `deployment/fmu/HVACUnitCoolerFMU.fmu` (2.4 MB)

#### 6. FMU Validation
```bash
python deployment/validation/validate_fmu_predictions.py
```
**What it does:**
- Tests FMU on 100 random test samples
- Compares FMU predictions vs real values
- **Expected:** R²=0.78-0.92 average

### Pipeline Flow Diagram

```
Raw Data (56K samples, 32 features)
    ↓
[Sprint 1: run_sprint1_pipeline_no_leakage.py]
    ↓ Preprocessing (100% retention)
    ↓ Feature Engineering (NO leakage: 39 features)
    ↓ Temporal Split (70/15/15)
    ↓ Scaling (StandardScaler)
    ↓
data/processed_no_leakage/
    ↓
[train_model_no_leakage.py]
    ↓ Train 3 LightGBM models
    ↓
models/lightgbm_model_no_leakage.pkl
    ↓
[run_sprint5_evaluation.py] (optional)
    ↓ Comprehensive evaluation
    ↓
results/*.csv
    ↓
[deployment/scripts/export_fmu_sensor_inputs.py]
    ↓ Build FMU
    ↓
deployment/fmu/HVACUnitCoolerFMU.fmu
    ↓
[deployment/validation/validate_fmu_predictions.py]
    ↓ Validate on real data
    ↓
✅ Production Ready
```

---

## Key Conventions

### Naming Conventions

#### Python Files
- **snake_case** for modules: `data_loader.py`, `feature_engineering.py`
- **Descriptive verbs:** `preprocess_*`, `engineer_*`, `export_*`, `validate_*`
- **Suffix for variants:** `*_no_leakage.py` (no data leakage versions)

#### Variables

**Sensor Variables (20 inputs - UPPERCASE with UC prefix):**
```python
# Water side
UCWIT   # Unit Cooler Water Inlet Temperature (°C)
UCWOT   # Unit Cooler Water Outlet Temperature (°C) - TARGET
UCWF    # Unit Cooler Water Flow Rate (L/min)
UCWP    # Unit Cooler Water Pressure (bar)

# Air side
UCAIT   # Unit Cooler Air Inlet Temperature (°C)
UCAOT   # Unit Cooler Air Outlet Temperature (°C) - TARGET
UCAIH   # Unit Cooler Air Inlet Humidity (%)
UCAF    # Unit Cooler Air Flow (m³/h) - TARGET

# Control/System
UCTSP   # Unit Cooler Temperature SetPoint (°C)
UCFMS   # Unit Cooler Fan Motor Speed (RPM)
UCFMV   # Unit Cooler Fan Motor Voltage (V)
UCSPD   # Unit Cooler Speed (%)
UCKV    # Unit Cooler Valve (%)
UCPV    # Unit Cooler Pump Valve (%)

# Chiller
CPDP    # Chiller Primary Discharge Pressure (bar)
CPSP    # Chiller Primary Suction Pressure (bar)
CPPR    # Chiller Primary Pump Rate (%)

# Ambient
AMBT    # Ambient Temperature (°C)
```

**Engineered Features (19 features - lowercase with descriptive names):**
```python
# Temperature deltas (5)
T_approach                # UCWIT - UCAIT
T_water_ambient_diff      # UCWIT - AMBT
T_air_ambient_diff        # UCAIT - AMBT
setpoint_inlet_diff       # UCTSP - UCAIT
setpoint_ambient_diff     # UCTSP - AMBT

# Mass flow/thermodynamic (6)
mdot_water               # Water mass flow rate (kg/s)
C_water                  # Water heat capacity (J/K)
Q_max_water              # Maximum water-side heat transfer (W)
P_fan_estimate           # Fan power estimate (W)
P_pump_estimate          # Pump power estimate (W)
P_total_estimate         # Total power estimate (W)

# Temporal (4)
time_index               # Sequential time index
cycle_hour               # Hour within cycle (0-23)
hour_sin                 # sin(2π * hour / 24)
hour_cos                 # cos(2π * hour / 24)

# Interaction (4)
T_water_x_flow           # UCWIT * UCWF
ambient_x_inlet          # AMBT * UCAIT
setpoint_x_flow          # UCTSP * UCWF
T_water_x_pressure       # UCWIT * UCWP
```

**Classes:**
```python
DataLoader              # UpperCamelCase
DataPreprocessor
PhysicsFeatureEngineer
BaselineModel
```

**Functions:**
```python
load_and_preprocess()           # verb_noun pattern
preprocess_unit_cooler_data()
engineer_features_no_leakage()
_handle_sensor_saturation()     # Private: leading underscore
```

### Code Style

- **Formatter:** Black (line length 100)
- **Linter:** Flake8
- **Docstrings:** Google style
- **Imports:** Absolute imports preferred

Example:
```python
from src.data.data_loader import DataLoader
from src.data.feature_engineering_no_leakage import PhysicsFeatureEngineerNoLeakage
```

### File Organization Principles

1. **Core logic in `src/`** - Reusable library modules
2. **Scripts in root** - Pipeline execution scripts (sprint1, sprint2, etc.)
3. **Deployment in `deployment/`** - Production artifacts and build scripts
4. **Tests in `tests/`** - pytest unit tests
5. **Results isolated** - CSV results in `results/`, plots in `plots/`

---

## Data Pipeline

### Critical Rule: NO Data Leakage

**NEVER use target variables (UCAOT, UCWOT, UCAF) in feature engineering!**

The original pipeline had data leakage. The current version ensures:
- ✅ All 39 features computable from 20 sensor inputs only
- ✅ StandardScaler fitted ONLY on training data
- ✅ Temporal split preserves chronological order (no shuffling)
- ✅ Validation/test data scaled with training statistics

### Feature Engineering Rules

When modifying features, ensure:
1. **No target dependency:** Features must NOT use UCAOT, UCWOT, or UCAF
2. **Production readiness:** All features computable in real-time from sensors
3. **Physics-informed:** Use domain knowledge (thermodynamics, HVAC principles)
4. **Validation:** Test that FMU can predict with sensor inputs only

**Good features:**
```python
T_approach = UCWIT - UCAIT                    # ✅ Only sensors
mdot_water = UCWF * (rho_water / 1000)        # ✅ Only sensors
P_fan_estimate = UCFMS * UCFMV * efficiency   # ✅ Only sensors
```

**Bad features (data leakage):**
```python
delta_T_air = UCAOT - UCAIT                   # ❌ Uses target UCAOT
Q_air = UCAF * rho_air * Cp_air * delta_T_air # ❌ Uses target UCAF
effectiveness = delta_T_air / T_approach      # ❌ Uses target UCAOT
```

### Data Processing Steps

```python
# 1. Load raw data
from src.data.data_loader import DataLoader
loader = DataLoader('data/raw/datos_combinados_entrenamiento_20251118_105234.csv')
df = loader.load_and_preprocess()

# 2. Preprocess
from src.data.preprocessing import DataPreprocessor
preprocessor = DataPreprocessor()
df_clean = preprocessor.preprocess_unit_cooler_data(df)

# 3. Feature engineering (NO leakage)
from src.data.feature_engineering_no_leakage import PhysicsFeatureEngineerNoLeakage
engineer = PhysicsFeatureEngineerNoLeakage()
df_features = engineer.engineer_features(df_clean)

# 4. Temporal split (70/15/15)
from src.data.data_splits import TemporalDataSplitter
splitter = TemporalDataSplitter()
X_train, X_val, X_test, y_train, y_val, y_test = splitter.prepare_temporal_data(df_features)

# 5. Scale (fit on train only)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)    # Transform only
X_test_scaled = scaler.transform(X_test)  # Transform only
```

### Data Artifacts

After Sprint 1, you should have:
```
data/processed_no_leakage/
├── X_train_scaled.npy          # (39,347 × 39)
├── X_val_scaled.npy            # (8,431 × 39)
├── X_test_scaled.npy           # (8,433 × 39)
├── y_train_scaled.npy          # (39,347 × 3)
├── y_val_scaled.npy            # (8,431 × 3)
├── y_test_scaled.npy           # (8,433 × 3)
├── scaler.pkl                  # Input StandardScaler
├── y_scaler_clean.pkl          # Output StandardScaler
└── metadata.json               # Feature/target names
```

---

## Testing Guidelines

### Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test
pytest tests/test_data_pipeline.py::TestFeatureEngineering::test_temperature_deltas
```

### Test Coverage

Tests are in `tests/test_data_pipeline.py`:
- `TestDataLoader` - CSV loading
- `TestDataPreprocessor` - Cleaning, saturation, outliers
- `TestFeatureEngineering` - All 19 engineered features
- `TestDataSplitter` - Temporal splits
- `TestScaler` - Scaling/inverse scaling

### FMU Validation

**Critical:** Always validate FMU after building:
```bash
python deployment/validation/validate_fmu_predictions.py
```

**Expected results:**
- UCAOT: R² > 0.90
- UCWOT: R² > 0.75
- UCAF: R² > 0.65
- Average: R² > 0.75

If validation fails, check:
1. Model training completed successfully
2. Scalers are correct (input + output)
3. Feature engineering matches training pipeline
4. FMU resource files are included

### Adding New Tests

When modifying code, add tests:
```python
# tests/test_data_pipeline.py
def test_new_feature():
    """Test description."""
    engineer = PhysicsFeatureEngineerNoLeakage()
    df = engineer.engineer_features(sample_df)
    assert 'new_feature' in df.columns
    assert df['new_feature'].dtype == float
    assert not df['new_feature'].isnull().any()
```

---

## Deployment Procedures

### Building FMU

**Standard workflow:**
```bash
# 1. Ensure model is trained
python train_model_no_leakage.py

# 2. Extract output scaler
python deployment/scripts/extract_y_scaler_for_fmu.py

# 3. Clean model and scaler artifacts
python deployment/scripts/clean_model_for_fmu.py
python deployment/scripts/clean_scaler_for_fmu.py

# 4. Build FMU
python deployment/scripts/export_fmu_sensor_inputs.py

# 5. Validate FMU
python deployment/validation/validate_fmu_predictions.py
```

**Quick rebuild (if model unchanged):**
```bash
python deployment/scripts/export_fmu_sensor_inputs.py
python deployment/validation/validate_fmu_predictions.py
```

### FMU Usage

**Python (FMPy):**
```python
from fmpy import simulate_fmu

# Define sensor inputs (physical units)
sensor_inputs = {
    'UCWIT': 7.5,      # Water inlet temp (°C)
    'UCAIT': 25.0,     # Air inlet temp (°C)
    'UCWF': 15.0,      # Water flow (L/min)
    'AMBT': 22.0,      # Ambient temp (°C)
    'UCTSP': 21.0,     # Setpoint (°C)
    'UCWP': 2.5,       # Water pressure (bar)
    'UCAIH': 50.0,     # Air inlet humidity (%)
    'UCFMS': 1200.0,   # Fan speed (RPM)
    'UCFMV': 230.0,    # Fan voltage (V)
    'CPPR': 60.0,      # Pump rate (%)
    'CPDP': 10.0,      # Discharge pressure (bar)
    'CPSP': 3.0,       # Suction pressure (bar)
    'UCSPD': 75.0,     # Unit speed (%)
    'UCKV': 50.0,      # Valve (%)
    'UCPV': 60.0,      # Pump valve (%)
    # Add remaining sensors as needed
}

# Simulate FMU
result = simulate_fmu(
    'deployment/fmu/HVACUnitCoolerFMU.fmu',
    start_values=sensor_inputs,
    stop_time=1.0
)

# Get predictions (physical units)
UCAOT = result['UCAOT'][-1]  # Air outlet temp (°C)
UCWOT = result['UCWOT'][-1]  # Water outlet temp (°C)
UCAF = result['UCAF'][-1]    # Air flow (m³/h)

print(f"Predictions: UCAOT={UCAOT:.2f}°C, UCWOT={UCWOT:.2f}°C, UCAF={UCAF:.0f} m³/h")
```

### Alternative Deployment Options

**ONNX Export:**
```bash
python deployment/scripts/export_model_to_onnx.py
# Output: deployment/onnx/hvac_model.onnx
```

**FastAPI Server:**
```bash
cd api/
uvicorn main:app --reload
# Access: http://localhost:8000/docs
```

**MQTT Integration:**
```python
from integration.mqtt.mqtt_client import HVACMQTTClient
client = HVACMQTTClient(broker_host='localhost', broker_port=1883)
client.publish_prediction(UCAOT, UCWOT, UCAF)
```

---

## Common Tasks

### Task 1: Retrain Model with New Data

```bash
# 1. Place new CSV in data/raw/
cp new_data.csv data/raw/

# 2. Update data_loader.py if needed (file path)

# 3. Run full pipeline
python run_sprint1_pipeline_no_leakage.py
python train_model_no_leakage.py

# 4. Rebuild FMU
python deployment/scripts/export_fmu_sensor_inputs.py

# 5. Validate
python deployment/validation/validate_fmu_predictions.py
```

### Task 2: Add New Engineered Feature

**CRITICAL:** Ensure no data leakage!

```python
# 1. Edit src/data/feature_engineering_no_leakage.py
class PhysicsFeatureEngineerNoLeakage:
    def engineer_features(self, df):
        # ... existing code ...

        # Add new feature (NO targets: UCAOT, UCWOT, UCAF)
        df['new_feature'] = df['UCWIT'] * df['UCWF']  # ✅ Only sensors

        return df

# 2. Add test in tests/test_data_pipeline.py
def test_new_feature():
    engineer = PhysicsFeatureEngineerNoLeakage()
    df = engineer.engineer_features(sample_df)
    assert 'new_feature' in df.columns

# 3. Rerun pipeline
python run_sprint1_pipeline_no_leakage.py
python train_model_no_leakage.py

# 4. Validate no leakage
# - Check feature uses only sensor inputs
# - Rebuild FMU and validate
```

### Task 3: Evaluate Model Performance

```bash
# Comprehensive evaluation
python run_sprint5_evaluation.py

# Check results
ls results/
# - feature_importance_complete.csv
# - residual_statistics.csv
# - performance_by_conditions.csv
# - cross_validation_temporal.csv
```

### Task 4: Compare Different Models

```bash
# Run baseline comparison
python run_sprint2_baseline.py

# Check results/baseline_comparison.csv
# Models compared: Linear, RandomForest, LightGBM, XGBoost
```

### Task 5: Create Release Package

```bash
# Package test data
python scripts/analysis/package_test_data.py

# Package models
python scripts/analysis/package_files_for_download.py

# Packages created in deployment/packages/
ls deployment/packages/
# - test_data_package.zip (121 KB)
# - validation_data_package.zip (676 KB)
# - hvac_models_package.tar.gz
```

### Task 6: Monitor for Data Drift

```python
from monitoring.drift_detector import DriftDetector

detector = DriftDetector()
drift_report = detector.detect_drift(
    reference_data=X_train,
    current_data=X_new
)

if drift_report['significant_drift']:
    print("⚠️ Data drift detected - consider retraining")
```

---

## Critical Constraints

### 1. Data Leakage Prevention

**NEVER use target variables in feature engineering:**
```python
# ❌ BAD - Uses targets
df['delta_T_air'] = df['UCAOT'] - df['UCAIT']  # UCAOT is target!
df['Q_air'] = df['UCAF'] * rho_air * Cp_air    # UCAF is target!

# ✅ GOOD - Only sensors
df['T_approach'] = df['UCWIT'] - df['UCAIT']
df['mdot_water'] = df['UCWF'] * rho_water / 1000
```

### 2. Temporal Split (No Shuffling)

**NEVER shuffle data before splitting:**
```python
# ❌ BAD - Random shuffling breaks temporal order
from sklearn.model_selection import train_test_split
X_train, X_test = train_test_split(X, y, test_size=0.3, shuffle=True)

# ✅ GOOD - Temporal split
train_size = int(0.7 * len(df))
X_train = X[:train_size]
X_test = X[train_size:]
```

### 3. Scaler Fitting

**NEVER fit scaler on validation/test data:**
```python
# ❌ BAD - Leaks validation data statistics
scaler.fit(np.concatenate([X_train, X_val]))

# ✅ GOOD - Fit on training only
scaler.fit(X_train)
X_val_scaled = scaler.transform(X_val)
```

### 4. FMU Input Requirements

**FMU requires exactly 20 sensor inputs (no more, no less):**
- UCWIT, UCAIT, UCWF, UCWP, UCAIH, AMBT, UCTSP, UCFMS, UCFMV, CPPR, CPDP, CPSP, UCSPD, UCKV, UCPV
- + 5 more sensors (see sensor list in Key Conventions)

All 19 engineered features are computed internally by the FMU.

### 5. Target Variables

**Exactly 3 targets (do not add/remove):**
- UCAOT (Air Outlet Temperature - °C)
- UCWOT (Water Outlet Temperature - °C)
- UCAF (Air Flow - m³/h)

### 6. Model Architecture

**3 independent LightGBM models (one per target):**
```python
# ✅ GOOD - Multi-output approach
models = {
    'UCAOT': LGBMRegressor(),
    'UCWOT': LGBMRegressor(),
    'UCAF': LGBMRegressor()
}

# ❌ BAD - Single multi-output model
# (reduces flexibility for per-target tuning)
```

### 7. Performance Expectations

**Realistic R² ranges (validated on real data):**
- UCAOT: 0.90-0.93 (excellent)
- UCWOT: 0.75-0.80 (good)
- UCAF: 0.65-0.75 (acceptable)
- Average: 0.75-0.85 (production-ready)

**Red flags:**
- R² > 0.99 → likely data leakage!
- R² < 0.50 → model needs improvement

### 8. File Paths

**Always use processed_no_leakage directory:**
```python
# ✅ GOOD
data_path = 'data/processed_no_leakage/X_train_scaled.npy'

# ❌ BAD - Old directory with leakage
data_path = 'data/processed/X_train_scaled.npy'
```

---

## Git Workflow

### Branch Structure

- **Main branch:** `main` (production-ready code)
- **Feature branches:** `claude/*` pattern (e.g., `claude/claude-md-mia1fv9i6u3d0v5x-01HoJh2pydszBWYRxwbmxM7f`)
- **Pull requests:** Used for major changes (#3, #4, #5, #6)

### Commit Message Conventions

```
[type]: [description]

Types:
- feat:     New features
- fix:      Bug fixes
- refactor: Code reorganization
- docs:     Documentation updates
- chore:    Maintenance and utilities
- test:     Adding tests

Examples:
feat: Add new physics-based feature for thermal power
fix: Resolve data leakage in feature engineering
docs: Update README with FMU validation results
chore: Add FMU validation scripts
```

### .gitignore Important Exclusions

Large files excluded from git:
- `data/raw/*` - Raw CSV files (download separately)
- `data/processed_no_leakage/*` - Generated arrays (.gitkeep only)
- `models/*.pkl` - Trained models (too large)
- `results/*.csv` - Generated results
- `plots/**/*.png` - Generated plots

**To download data:**
```bash
python scripts/download_training_data.py
# OR download from deployment/packages/
```

### Making Changes

```bash
# 1. Ensure you're on feature branch
git status
# Current branch: claude/claude-md-mia1fv9i6u3d0v5x-01HoJh2pydszBWYRxwbmxM7f

# 2. Make changes and test
python run_sprint1_pipeline_no_leakage.py
pytest tests/

# 3. Stage and commit
git add src/data/feature_engineering_no_leakage.py
git commit -m "feat: Add new physics-based interaction feature"

# 4. Push to remote
git push -u origin claude/claude-md-mia1fv9i6u3d0v5x-01HoJh2pydszBWYRxwbmxM7f
```

### Pull Request Workflow

When making significant changes:
1. Create feature branch
2. Make changes with tests
3. Update documentation (README, CHANGELOG)
4. Create pull request
5. Review and merge

---

## Quick Reference

### Key Files

| File | Purpose |
|------|---------|
| `run_sprint1_pipeline_no_leakage.py` | Data preparation pipeline |
| `train_model_no_leakage.py` | Model training |
| `deployment/scripts/export_fmu_sensor_inputs.py` | Build FMU |
| `deployment/validation/validate_fmu_predictions.py` | Validate FMU |
| `src/data/feature_engineering_no_leakage.py` | Feature engineering (NO leakage) |
| `requirements.txt` | Python dependencies |
| `README.md` | Project documentation |
| `CHANGELOG_NO_LEAKAGE.md` | Development history |

### Key Commands

```bash
# Full pipeline
python run_sprint1_pipeline_no_leakage.py && \
python train_model_no_leakage.py && \
python deployment/scripts/export_fmu_sensor_inputs.py && \
python deployment/validation/validate_fmu_predictions.py

# Quick test
pytest tests/

# Rebuild FMU only
python deployment/scripts/export_fmu_sensor_inputs.py

# Validate FMU
python deployment/validation/validate_fmu_predictions.py
```

### Performance Metrics

| Metric | UCAOT | UCWOT | UCAF |
|--------|-------|-------|------|
| **R² (Validation)** | 0.913 | 0.747 | 0.754 |
| **MAE (Validation)** | 0.136 | 0.253 | 0.200 |
| **R² (FMU Test)** | 0.924 | 0.760 | 0.665 |
| **Status** | ✅ Excellent | ✅ Good | ✅ Acceptable |

---

## Documentation

### Available Docs

- **README.md** - Main project documentation (16 KB)
- **CHANGELOG_NO_LEAKAGE.md** - Development history (11 KB)
- **data/DATA_SUMMARY.md** - Dataset documentation
- **deployment/fmu/README_SENSOR_INPUTS.md** - FMU usage guide
- **deployment/fmu/FMU_SETUP_GUIDE.md** - FMU setup instructions
- **docs/Sprint*.md** - Sprint-specific documentation (Sprints 0-8)
- **docs/final/** - Final deliverables (project summary, user manual, maintenance guide)

### External Resources

- **FMI Standard:** https://fmi-standard.org/ (FMI 2.0 Co-Simulation)
- **LightGBM Docs:** https://lightgbm.readthedocs.io/
- **FMPy Library:** https://github.com/CATIA-Systems/FMPy

---

## Troubleshooting

### Issue: FMU validation fails with low R²

**Solution:**
1. Check model training completed: `ls models/lightgbm_model_no_leakage.pkl`
2. Verify scalers exist: `ls data/processed_no_leakage/*.pkl`
3. Rebuild FMU: `python deployment/scripts/export_fmu_sensor_inputs.py`
4. Re-validate: `python deployment/validation/validate_fmu_predictions.py`

### Issue: Data leakage suspected (R² > 0.99)

**Solution:**
1. Check feature engineering: `grep -n "UCAOT\|UCWOT\|UCAF" src/data/feature_engineering_no_leakage.py`
2. Ensure no target variables used in features
3. Use `run_sprint1_pipeline_no_leakage.py` (NOT `run_sprint1_pipeline.py`)

### Issue: Import errors

**Solution:**
```bash
# Add src to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Or use absolute imports
from src.data.data_loader import DataLoader
```

### Issue: Missing dependencies

**Solution:**
```bash
pip install -r requirements.txt

# For FMU support
pip install pythonfmu fmpy

# For integration
pip install -r requirements.integration.txt
```

---

## Best Practices for AI Assistants

### When Working with This Codebase

1. **Always check for data leakage** when modifying features
2. **Run tests** after code changes: `pytest tests/`
3. **Validate FMU** after rebuilding: `python deployment/validation/validate_fmu_predictions.py`
4. **Update documentation** when making significant changes (README, CHANGELOG)
5. **Use temporal splits** - never shuffle time-series data
6. **Fit scalers on training data only** - transform val/test
7. **Follow naming conventions** - sensors UPPERCASE, features lowercase
8. **Add tests** for new features or modifications

### Before Suggesting Changes

1. **Read relevant source files** in `src/`
2. **Check if similar code exists** to maintain consistency
3. **Verify no data leakage** in proposed features
4. **Consider production impact** - can FMU compute this feature?
5. **Test proposed changes** with provided test infrastructure

### When Debugging

1. **Check logs** from pipeline scripts
2. **Verify file paths** match processed_no_leakage directories
3. **Inspect data shapes** - X should be (N, 39), y should be (N, 3)
4. **Compare metrics** against expected performance
5. **Review FMU validation** output for discrepancies

---

## Summary

This is a **production-ready digital twin** with:
- ✅ **Zero data leakage** (all features from sensors only)
- ✅ **Realistic performance** (R²=0.78-0.92 on real data)
- ✅ **FMU deployment** (2.4 MB, <1ms inference)
- ✅ **Comprehensive testing** (pytest + FMU validation)
- ✅ **Well-documented** (21 markdown files)
- ✅ **Modular architecture** (src/ for library, deployment/ for production)

**Key takeaway:** This project solved a critical data leakage problem. Always verify that features use ONLY sensor inputs, never target variables.

---

**For questions or issues:**
1. Check documentation first (README, CHANGELOG, Sprint docs)
2. Run validation scripts
3. Review error messages carefully
4. Contact: [rferreiroag](https://github.com/rferreiroag)

**Version:** 1.0.0
**Last Updated:** 2025-11-22
**Status:** ✅ Production Ready
