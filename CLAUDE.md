# CLAUDE.md - AI Assistant Guide for Unit Cooler CIS

**Last Updated:** 2025-11-22
**Project Status:** Production Ready
**Version:** 1.0.0

This document provides comprehensive guidance for AI assistants working with the HVAC Unit Cooler Digital Twin codebase. It explains the project structure, development workflows, key conventions, and best practices to follow.

---

## 📋 Table of Contents

1. [Project Overview](#project-overview)
2. [Repository Structure](#repository-structure)
3. [Development Workflow](#development-workflow)
4. [Key Principles & Conventions](#key-principles--conventions)
5. [Data Pipeline](#data-pipeline)
6. [Model Training](#model-training)
7. [FMU Deployment](#fmu-deployment)
8. [Testing & Validation](#testing--validation)
9. [Common Tasks](#common-tasks)
10. [Code Patterns](#code-patterns)
11. [Git Workflow](#git-workflow)
12. [Troubleshooting](#troubleshooting)

---

## 🎯 Project Overview

### Purpose

This project implements a **Production-Ready Data-Driven Digital Twin** for HVAC Unit Cooler systems using LightGBM models with physics-based feature engineering. The system predicts Unit Cooler outputs (air outlet temperature, water outlet temperature, and air flow) from 20 sensor inputs.

### Key Achievements

- **Performance:** R²=0.78-0.92 on real test data
- **Data Quality:** NO data leakage - production ready
- **Deployment:** FMI 2.0 Co-Simulation FMU (2.4 MB)
- **Inference:** <1ms prediction time
- **Test Coverage:** 8,432 real test samples validated

### Core Technologies

- **ML Framework:** LightGBM 4.1+
- **Language:** Python 3.11+
- **Data Processing:** NumPy, Pandas, Scikit-learn
- **Deployment:** PythonFMU (FMI 2.0)
- **Validation:** FMPy

---

## 📂 Repository Structure

```
Unit_Cooler_CIS/
├── data/                                   # All datasets
│   ├── raw/                               # Original CSV data (gitignored)
│   ├── processed/                         # OLD pipeline (with leakage)
│   └── processed_no_leakage/              # ✅ PRODUCTION data (use this!)
│       ├── X_train_scaled.npy             # Training features (39,347 × 39)
│       ├── X_val_scaled.npy               # Validation features
│       ├── X_test_scaled.npy              # Test features (8,432 × 39)
│       ├── y_train_scaled.npy             # Training targets
│       ├── y_val_scaled.npy               # Validation targets
│       ├── y_test_scaled.npy              # Test targets (8,432 × 3)
│       ├── scaler.pkl                     # Input StandardScaler
│       ├── y_scaler_clean.pkl             # Output StandardScaler
│       └── metadata.json                  # Feature/target names
│
├── src/                                   # Source code modules
│   ├── data/                             # Data processing
│   │   ├── data_loader.py                # Load raw CSV data
│   │   ├── preprocessing.py              # Clean & preprocess
│   │   ├── data_splits.py                # Temporal train/val/test split
│   │   ├── feature_engineering.py        # OLD (with leakage - don't use!)
│   │   └── feature_engineering_no_leakage.py  # ✅ PRODUCTION (use this!)
│   ├── models/                           # Model implementations
│   │   ├── baseline_models.py            # LightGBM, XGBoost
│   │   ├── advanced_models.py            # MLP, ensembles
│   │   └── pinn_model.py                 # Physics-informed NN (experimental)
│   ├── utils/                            # Utilities
│   │   ├── eda_utils.py                  # Exploratory data analysis
│   │   ├── visualization.py              # Plotting functions
│   │   └── feature_analysis.py           # Feature importance analysis
│   ├── losses/                           # Custom loss functions
│   └── optimization/                     # Hyperparameter optimization
│
├── models/                                # Trained models (gitignored)
│   ├── lightgbm_model_no_leakage.pkl     # ✅ PRODUCTION model (1.8 MB)
│   └── lightgbm_model_no_leakage_clean.pkl  # FMU-ready model
│
├── deployment/                            # Deployment artifacts
│   ├── fmu/                              # FMU files
│   │   ├── HVACUnitCoolerFMU.fmu         # ⭐ PRODUCTION FMU (2.4 MB)
│   │   ├── hvac_fmu_sensor_inputs.py     # FMU implementation
│   │   ├── resources/                     # FMU internal resources
│   │   ├── README_SENSOR_INPUTS.md        # FMU usage guide
│   │   ├── FMU_SETUP_GUIDE.md            # Setup instructions
│   │   └── REQUIREMENTS_FMU.txt           # FMU dependencies
│   ├── scripts/                          # Build scripts
│   │   ├── export_fmu_sensor_inputs.py   # ⭐ Build FMU from model
│   │   ├── extract_y_scaler_for_fmu.py   # Extract output scaler
│   │   ├── clean_model_for_fmu.py        # Clean model artifacts
│   │   ├── clean_scaler_for_fmu.py       # Clean scaler artifacts
│   │   └── export_model_to_onnx.py       # ONNX export (optional)
│   ├── validation/                       # FMU testing
│   │   ├── validate_fmu_predictions.py   # ⭐ Validate FMU (100 samples)
│   │   ├── test_fmu.py                   # Quick FMU test
│   │   ├── test_fmu_comprehensive.py     # Full FMU test
│   │   └── example_inference.py          # Usage examples
│   ├── packages/                         # Release packages
│   │   ├── test_data_package.zip         # Test data (121 KB)
│   │   ├── validation_data_package.zip   # Validation data (676 KB)
│   │   └── hvac_models_package.tar.gz    # Models package
│   ├── onnx/                             # ONNX deployment (optional)
│   ├── docker/                           # Docker containers (optional)
│   └── benchmarks/                       # Performance tests
│
├── scripts/                               # Utility scripts
│   ├── analysis/                         # Data analysis
│   │   ├── investigate_validation_data.py    # Data source analysis
│   │   ├── analyze_test_data_detail.py       # Test set analysis
│   │   ├── debug_test_data.py                # Test data debugging
│   │   ├── package_test_data.py              # Package creator
│   │   ├── package_files_for_download.py     # File packager
│   │   └── generate_sprint2_analysis.py      # Sprint 2 analysis
│   └── download_training_data.py         # Data downloader
│
├── tests/                                 # Unit tests
│   └── test_data_pipeline.py             # Data pipeline tests
│
├── api/                                   # REST API (optional)
│   └── main.py                           # FastAPI implementation
│
├── integration/                           # Integration adapters
│   ├── mqtt/                             # MQTT integration
│   └── bacnet/                           # BACnet integration
│
├── monitoring/                            # Model monitoring
│   └── drift_report.json                 # Data drift reports
│
├── dashboard/                             # Streamlit dashboard (optional)
├── notebooks/                             # Jupyter notebooks
├── plots/                                 # Generated plots (gitignored)
├── results/                               # Analysis results (gitignored)
├── docs/                                  # Sprint documentation
│
├── run_sprint1_pipeline_no_leakage.py    # ⭐ Data preparation
├── train_model_no_leakage.py             # ⭐ Model training
├── run_sprint2_baseline.py               # Baseline comparison
├── run_sprint3_pinn.py                   # PINN experiments
├── run_sprint5_evaluation.py             # Comprehensive evaluation
├── run_sprint6_deployment.py             # Deployment pipeline
│
├── requirements.txt                       # Main dependencies
├── requirements.edge.txt                  # Edge deployment deps
├── requirements.integration.txt           # Integration deps
├── .gitignore                            # Git ignore rules
├── README.md                             # Main documentation
├── CHANGELOG_NO_LEAKAGE.md               # Change history
└── CLAUDE.md                             # This file
```

### File Organization Principles

**IMPORTANT:** This repository has TWO pipelines:

1. **OLD Pipeline** (with data leakage - DO NOT USE):
   - `src/data/feature_engineering.py`
   - `data/processed/`
   - These files create features that depend on target variables

2. **PRODUCTION Pipeline** (NO leakage - ALWAYS USE):
   - `src/data/feature_engineering_no_leakage.py`
   - `data/processed_no_leakage/`
   - All features computable from 20 sensor inputs only

**Rule:** Always use the `_no_leakage` variants for any new work!

---

## 🔄 Development Workflow

### Sprint-Based Development

The project follows a sprint-based methodology:

1. **Sprint 1:** Data preparation & feature engineering (no leakage)
2. **Sprint 2:** Baseline model development (LightGBM, XGBoost)
3. **Sprint 3:** Advanced models (PINN experiments)
4. **Sprint 5:** Comprehensive evaluation
5. **Sprint 6:** FMU deployment & validation
6. **Sprint 7:** Real-time integration (MQTT, BACnet)
7. **Sprint 8:** Documentation & knowledge transfer

### Standard Development Cycle

```bash
# 1. Data Preparation (if needed)
python run_sprint1_pipeline_no_leakage.py

# 2. Model Training
python train_model_no_leakage.py

# 3. Evaluation
python run_sprint5_evaluation.py

# 4. FMU Export
python deployment/scripts/export_fmu_sensor_inputs.py

# 5. Validation
python deployment/validation/validate_fmu_predictions.py
```

---

## 🔑 Key Principles & Conventions

### 1. NO Data Leakage Principle

**CRITICAL:** The most important principle in this project!

**Problem:**
- Original pipeline used target variables (UCAOT, UCWOT, UCAF) to create features
- Example: `delta_T_air = UCAOT - UCAIT` (uses UCAOT which we're trying to predict!)
- This caused artificially high R² (>0.99) that couldn't be replicated in production

**Solution:**
- All features MUST be computable from **20 sensor inputs only**
- NO dependency on target variables
- See `src/data/feature_engineering_no_leakage.py` for correct implementation

**Validation:**
```python
# All 39 features come from:
# - 20 raw sensors (UCWIT, UCAIT, UCWF, etc.)
# - 19 derived features (computed from sensors only)
#
# NEVER use UCAOT, UCWOT, or UCAF in feature engineering!
```

### 2. File Naming Conventions

- **Scripts:** `verb_noun.py` (e.g., `train_model.py`, `export_fmu.py`)
- **Modules:** `noun.py` (e.g., `preprocessing.py`, `visualization.py`)
- **Test files:** `test_*.py` (e.g., `test_data_pipeline.py`)
- **Analysis scripts:** `analyze_*.py` or `investigate_*.py`
- **Package scripts:** `package_*.py`
- **No leakage variants:** Add `_no_leakage` suffix (e.g., `feature_engineering_no_leakage.py`)

### 3. Data File Conventions

- **NumPy arrays:** `.npy` extension (e.g., `X_train_scaled.npy`)
- **Models:** `.pkl` extension (joblib serialization)
- **Scalers:** `.pkl` extension (joblib serialization)
- **Metadata:** `.json` extension
- **Raw data:** `.csv` extension (in `data/raw/`)
- **Packages:** `.zip` or `.tar.gz` (in `deployment/packages/`)

### 4. Naming Standards

**Variables (20 Sensors):**
- UCWIT - Unit Cooler Water Inlet Temperature
- UCAIT - Unit Cooler Air Inlet Temperature
- UCWF - Unit Cooler Water Flow
- AMBT - Ambient Temperature
- UCTSP - Unit Cooler Temperature Setpoint
- UCWP - Unit Cooler Water Pressure
- CPDP - Compressor Discharge Pressure
- CPSP - Compressor Suction Pressure
- CPPR - Compressor Power
- UCFMS - Unit Cooler Fan Motor Speed
- UCFMV - Unit Cooler Fan Motor Voltage
- UCFS - Unit Cooler Fan Status
- CPCF - Compressor Cooling Flow
- MVWF - Mixing Valve Water Flow
- CPMHP - Compressor Motor HP
- CPMEP - Compressor Motor Electric Power
- CPMMP - Compressor Motor Max Power
- CPHE - Compressor Hours
- CPOC - Compressor On/Off Count
- CPES - Compressor Error Status

**Targets (3 Predictions):**
- UCAOT - Unit Cooler Air Outlet Temperature
- UCWOT - Unit Cooler Water Outlet Temperature
- UCAF - Unit Cooler Air Flow

### 5. Code Style

- **Docstrings:** Use triple-quoted docstrings for all modules and functions
- **Comments:** Explain WHY, not WHAT
- **Line length:** Aim for 80-100 characters
- **Imports:** Standard library → Third-party → Local modules
- **Print statements:** Use descriptive messages with `[step/total]` format

Example:
```python
print("="*80)
print(" TRAINING MODEL WITHOUT DATA LEAKAGE")
print("="*80)
print("\n[1/4] Loading data...")
print(f"✓ Data loaded")
```

---

## 📊 Data Pipeline

### Overview

```
Raw CSV Data (56,211 samples × 32 features)
    ↓
Preprocessing (100% retention)
    ↓
Feature Engineering (39 features total)
    ↓ [Remove any target dependencies]
Temporal Split (70% / 15% / 15%)
    ↓
Scaling (StandardScaler)
    ↓
Train / Validation / Test Sets
```

### Running the Data Pipeline

```bash
# Run complete data preparation
python run_sprint1_pipeline_no_leakage.py

# Outputs to: data/processed_no_leakage/
# - X_train_scaled.npy, y_train_scaled.npy
# - X_val_scaled.npy, y_val_scaled.npy
# - X_test_scaled.npy, y_test_scaled.npy
# - scaler.pkl, y_scaler_clean.pkl
# - metadata.json
```

### Feature Engineering Details

**Input:** 20 sensor measurements (raw features)

**Output:** 39 features total
- 20 raw sensor features (unchanged)
- 19 derived physics-based features

**Derived Features (19):**

1. **Temperature Features (5):**
   - T_approach = UCWIT - AMBT
   - T_water_ambient_diff = UCWIT - AMBT
   - T_air_ambient_diff = UCAIT - AMBT
   - setpoint_inlet_diff = UCTSP - UCAIT
   - setpoint_ambient_diff = UCTSP - AMBT

2. **Thermodynamic Features (6):**
   - mdot_water = UCWF * rho_water / 60  (kg/s)
   - C_water = mdot_water * cp_water
   - Q_max_water = C_water * abs(UCWIT - AMBT)
   - P_fan_estimate = UCFMV * (UCFMS / 3000)
   - P_pump_estimate = UCWP * UCWF / 60
   - P_total_estimate = P_fan_estimate + P_pump_estimate + CPPR

3. **Temporal Features (4):**
   - time_index = sequential index
   - cycle_hour = hour of day (0-23)
   - hour_sin = sin(2π * hour / 24)
   - hour_cos = cos(2π * hour / 24)

4. **Interaction Features (4):**
   - T_water_x_flow = UCWIT * UCWF
   - ambient_x_inlet = AMBT * UCAIT
   - setpoint_x_flow = UCTSP * UCWF
   - T_water_x_pressure = UCWIT * UCWP

**Key Point:** ALL features are computable from sensors only!

### Data Split Strategy

**Method:** Temporal split (preserves time ordering)

```python
# Split ratios
train: 70% (first chronologically)
val:   15% (middle)
test:  15% (last - most recent data)

# Why temporal?
# - Respects time-series nature
# - Tests generalization to future data
# - Prevents leakage from future to past
```

### Scaling

```python
# Input features: StandardScaler (zero mean, unit variance)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Output targets: StandardScaler (same approach)
y_scaler = StandardScaler()
y_scaled = y_scaler.fit_transform(y)

# Important: Fit on training data only!
# Transform validation/test using training statistics
```

---

## 🤖 Model Training

### Quick Training

```bash
# Train production model
python train_model_no_leakage.py

# Outputs:
# - models/lightgbm_model_no_leakage.pkl (1.8 MB)
# - Console output with validation metrics
```

### Model Architecture

**Algorithm:** LightGBM Gradient Boosting Decision Trees

**Strategy:** Multi-output regression (3 independent models)
- Model 1: Predicts UCAOT (air outlet temperature)
- Model 2: Predicts UCWOT (water outlet temperature)
- Model 3: Predicts UCAF (air flow)

**Hyperparameters:**
```python
lgbm_params = {
    'n_estimators': 200,        # Number of boosting rounds
    'max_depth': 8,             # Maximum tree depth
    'learning_rate': 0.05,      # Step size shrinkage
    'num_leaves': 31,           # Max leaves per tree
    'min_child_samples': 20,    # Min samples per leaf
    'subsample': 0.8,           # Row sampling ratio
    'colsample_bytree': 0.8,    # Column sampling ratio
    'random_state': 42,         # Reproducibility
    'n_jobs': -1,               # Use all CPU cores
    'verbose': -1               # Suppress warnings
}
```

### Training Process

```python
# For each target:
for i, target in enumerate(['UCAOT', 'UCWOT', 'UCAF']):
    # 1. Initialize model
    model = LGBMRegressor(**lgbm_params)

    # 2. Train on scaled data
    model.fit(X_train_scaled, y_train_scaled[:, i])

    # 3. Validate
    y_pred = model.predict(X_val_scaled)
    r2 = r2_score(y_val_scaled[:, i], y_pred)

    # 4. Store model
    models[target] = model
```

### Expected Performance (Validation Set)

| Target | R² | MAE | RMSE | Status |
|--------|-----|-----|------|--------|
| UCAOT | 0.913 | 0.136 | 0.224 | ⭐ Excellent |
| UCWOT | 0.747 | 0.253 | 0.515 | ✅ Good |
| UCAF | 0.754 | 0.200 | 0.472 | ✅ Good |
| **Average** | **0.805** | - | - | ✅ **Production Ready** |

**Note:** These are scaled values. Physical units have different ranges.

### Model Files

```
models/
├── lightgbm_model_no_leakage.pkl        # Main model (3 LightGBM models)
└── lightgbm_model_no_leakage_clean.pkl  # FMU-ready (no unnecessary deps)
```

**Loading models:**
```python
import joblib

# Load model dictionary
models = joblib.load('models/lightgbm_model_no_leakage.pkl')

# Access individual models
ucaot_model = models['UCAOT']
ucwot_model = models['UCWOT']
ucaf_model = models['UCAF']

# Make predictions
predictions = {
    'UCAOT': ucaot_model.predict(X),
    'UCWOT': ucwot_model.predict(X),
    'UCAF': ucaf_model.predict(X)
}
```

---

## 📦 FMU Deployment

### FMU Overview

**FMU = Functional Mock-up Unit**

A standardized format (FMI 2.0) for model exchange and co-simulation. Our FMU packages the trained LightGBM model into a portable unit that can be used in:
- Modelica/Dymola
- MATLAB/Simulink
- Python (FMPy)
- OpenModelica
- Building automation systems

### Building the FMU

```bash
# Build production FMU
cd deployment/scripts
python export_fmu_sensor_inputs.py

# Output: deployment/fmu/HVACUnitCoolerFMU.fmu (2.4 MB)
# Build time: ~10 seconds
```

### FMU Specifications

| Specification | Value |
|---------------|-------|
| Standard | FMI 2.0 Co-Simulation |
| Size | 2.4 MB |
| Inputs | 20 sensors (physical units) |
| Outputs | 3 predictions (physical units) |
| Internal Features | 39 (20 sensors + 19 computed) |
| Models | 3 LightGBM models |
| Inference Time | <1 ms |
| Data Leakage | None ✅ |

### FMU Internal Process

```
User provides: 20 sensor values (physical units)
    ↓
[Inside FMU]
    ↓
1. Input Scaling (StandardScaler)
    ↓
2. Feature Engineering (19 features)
    ↓
3. LightGBM Prediction (3 models)
    ↓
4. Output Descaling (StandardScaler⁻¹)
    ↓
Return: 3 predictions (physical units)
```

### Using the FMU (Python)

```python
from fmpy import simulate_fmu

# Define sensor inputs (physical values)
sensor_inputs = {
    'UCWIT': 7.5,      # Water inlet temp (°C)
    'UCAIT': 25.0,     # Air inlet temp (°C)
    'UCWF': 15.0,      # Water flow (L/min)
    'AMBT': 22.0,      # Ambient temp (°C)
    'UCTSP': 21.0,     # Setpoint (°C)
    'UCWP': 2.5,       # Water pressure (bar)
    'CPDP': 8.5,       # Compressor discharge pressure (bar)
    'CPSP': 4.2,       # Compressor suction pressure (bar)
    'CPPR': 1500,      # Compressor power (W)
    'UCFMS': 2400,     # Fan speed (RPM)
    'UCFMV': 230,      # Fan voltage (V)
    'UCFS': 1,         # Fan status (0/1)
    'CPCF': 12.0,      # Compressor flow (L/min)
    'MVWF': 10.0,      # Mixing valve flow (L/min)
    'CPMHP': 2.0,      # Compressor motor HP
    'CPMEP': 1200,     # Motor electric power (W)
    'CPMMP': 2000,     # Motor max power (W)
    'CPHE': 1000,      # Compressor hours
    'CPOC': 50,        # On/off count
    'CPES': 0          # Error status
}

# Simulate FMU
result = simulate_fmu(
    'deployment/fmu/HVACUnitCoolerFMU.fmu',
    start_values=sensor_inputs,
    stop_time=1.0
)

# Extract predictions (physical values)
UCAOT = result['UCAOT'][-1]  # Air outlet temp (°C)
UCWOT = result['UCWOT'][-1]  # Water outlet temp (°C)
UCAF = result['UCAF'][-1]    # Air flow (m³/h)

print(f"Predictions:")
print(f"  Air Outlet Temp: {UCAOT:.2f}°C")
print(f"  Water Outlet Temp: {UCWOT:.2f}°C")
print(f"  Air Flow: {UCAF:.0f} m³/h")
```

### FMU Files Structure

```
deployment/fmu/
├── HVACUnitCoolerFMU.fmu              # Main FMU file
├── hvac_fmu_sensor_inputs.py          # FMU implementation
├── resources/                          # Internal resources
│   ├── lightgbm_model_no_leakage_clean.pkl
│   ├── scaler_clean.pkl
│   └── y_scaler_clean.pkl
├── README_SENSOR_INPUTS.md            # Usage guide
├── FMU_SETUP_GUIDE.md                 # Setup instructions
└── REQUIREMENTS_FMU.txt                # Dependencies
```

---

## ✅ Testing & Validation

### Test Structure

```
tests/
├── test_data_pipeline.py              # Data pipeline unit tests

deployment/validation/
├── validate_fmu_predictions.py        # ⭐ Main FMU validation
├── test_fmu.py                        # Quick FMU functionality test
├── test_fmu_comprehensive.py          # Comprehensive FMU test
└── example_inference.py               # Usage examples
```

### Running Tests

```bash
# 1. Unit tests
cd tests
pytest test_data_pipeline.py

# 2. FMU validation (IMPORTANT!)
cd deployment/validation
python validate_fmu_predictions.py

# Expected output:
# ✓ UCAOT: R²=0.92, MAE=1.75°C
# ✓ UCWOT: R²=0.76, MAE=15.51°C
# ✓ UCAF: R²=0.67, MAE=340.86 m³/h
# ✓ Average R²=0.78 (PASS)
```

### FMU Validation Process

**Script:** `deployment/validation/validate_fmu_predictions.py`

**What it does:**
1. Loads 8,432 test samples (scaled)
2. Unscales to physical values
3. Selects 100 random samples
4. Simulates FMU with physical sensor values
5. Compares FMU predictions vs ground truth
6. Calculates metrics (R², MAE, RMSE, MAPE)

**Why it's important:**
- Validates that FMU works with real data
- Ensures scaling/descaling works correctly
- Confirms no data leakage in production
- Tests end-to-end inference pipeline

### Expected FMU Performance

| Variable | R² | MAE | RMSE | Interpretation |
|----------|-----|-----|------|----------------|
| UCAOT | 0.924 | 1.75°C | 2.42°C | ⭐ Excellent |
| UCWOT | 0.760 | 15.51°C | 19.25°C | ✅ Good |
| UCAF | 0.665 | 340.86 m³/h | 884.33 m³/h | ✅ Acceptable |
| **Average** | **0.783** | - | - | ✅ **Production Ready** |

**Acceptance Criteria:** Average R² > 0.75

---

## 🛠️ Common Tasks

### Task 1: Retrain Model with New Data

```bash
# 1. Place new CSV in data/raw/
cp new_data.csv data/raw/datos_combinados_entrenamiento_YYYYMMDD_HHMMSS.csv

# 2. Update data loader (if needed)
# Edit: src/data/data_loader.py

# 3. Run data pipeline
python run_sprint1_pipeline_no_leakage.py

# 4. Train model
python train_model_no_leakage.py

# 5. Validate
python run_sprint5_evaluation.py

# 6. Rebuild FMU
python deployment/scripts/export_fmu_sensor_inputs.py

# 7. Validate FMU
python deployment/validation/validate_fmu_predictions.py
```

### Task 2: Add New Feature

**CRITICAL:** Ensure NO data leakage!

```python
# Edit: src/data/feature_engineering_no_leakage.py

def engineer_features_no_leakage(df):
    """
    Engineer features WITHOUT using target variables
    """
    df = df.copy()

    # ... existing features ...

    # NEW FEATURE - Example: heat transfer coefficient estimate
    # ✅ GOOD: Uses only sensor inputs
    df['h_estimate'] = df['UCWF'] * df['UCWP'] / (df['UCWIT'] + 273.15)

    # ❌ BAD: Uses target variable!
    # df['delta_T_air'] = df['UCAOT'] - df['UCAIT']  # DON'T DO THIS!

    return df

# Then update metadata
feature_names = [
    # ... existing features ...
    'h_estimate'  # Add new feature name
]
```

**Then rerun pipeline:**
```bash
python run_sprint1_pipeline_no_leakage.py
python train_model_no_leakage.py
python deployment/scripts/export_fmu_sensor_inputs.py
python deployment/validation/validate_fmu_predictions.py
```

### Task 3: Export Model to Different Format

```bash
# ONNX export (for edge devices)
python deployment/scripts/export_model_to_onnx.py

# Output: deployment/onnx/hvac_model.onnx
```

### Task 4: Create Downloadable Package

```bash
# Package test data
python scripts/analysis/package_test_data.py
# Output: deployment/packages/test_data_package.zip

# Package validation data
python scripts/analysis/package_files_for_download.py
# Output: deployment/packages/validation_data_package.zip
```

### Task 5: Analyze Feature Importance

```bash
# Run comprehensive evaluation
python run_sprint5_evaluation.py

# Check outputs:
# - results/feature_importance_complete.csv
# - plots/sprint5/feature_importance_top20.png
```

### Task 6: Debug FMU Issues

```bash
# 1. Quick functionality test
python deployment/validation/test_fmu.py

# 2. If issues, check:
# - FMU file exists: deployment/fmu/HVACUnitCoolerFMU.fmu
# - Model exists: models/lightgbm_model_no_leakage_clean.pkl
# - Scalers exist: data/processed_no_leakage/scaler_clean.pkl
#                  data/processed_no_leakage/y_scaler_clean.pkl

# 3. Rebuild FMU
python deployment/scripts/export_fmu_sensor_inputs.py

# 4. Validate
python deployment/validation/validate_fmu_predictions.py
```

---

## 💻 Code Patterns

### Pattern 1: Loading Processed Data

```python
import numpy as np
import joblib
import json

# Load arrays
X_train = np.load('data/processed_no_leakage/X_train_scaled.npy')
y_train = np.load('data/processed_no_leakage/y_train_scaled.npy')
X_test = np.load('data/processed_no_leakage/X_test_scaled.npy')
y_test = np.load('data/processed_no_leakage/y_test_scaled.npy')

# Load scalers
scaler_X = joblib.load('data/processed_no_leakage/scaler.pkl')
scaler_y = joblib.load('data/processed_no_leakage/y_scaler_clean.pkl')

# Load metadata
with open('data/processed_no_leakage/metadata.json', 'r') as f:
    metadata = json.load(f)

feature_names = metadata['feature_names']  # List of 39 feature names
target_names = metadata['target_names']    # ['UCAOT', 'UCWOT', 'UCAF']
```

### Pattern 2: Making Predictions (Scaled Data)

```python
import joblib
import numpy as np

# Load model and scalers
models = joblib.load('models/lightgbm_model_no_leakage.pkl')
scaler_X = joblib.load('data/processed_no_leakage/scaler.pkl')
scaler_y = joblib.load('data/processed_no_leakage/y_scaler_clean.pkl')

# Load scaled test data
X_test_scaled = np.load('data/processed_no_leakage/X_test_scaled.npy')

# Predict (scaled outputs)
y_pred_scaled = np.zeros((len(X_test_scaled), 3))
for i, target in enumerate(['UCAOT', 'UCWOT', 'UCAF']):
    y_pred_scaled[:, i] = models[target].predict(X_test_scaled)

# Unscale to physical values
y_pred_physical = scaler_y.inverse_transform(y_pred_scaled)

print(f"UCAOT: {y_pred_physical[0, 0]:.2f}°C")
print(f"UCWOT: {y_pred_physical[0, 1]:.2f}°C")
print(f"UCAF: {y_pred_physical[0, 2]:.0f} m³/h")
```

### Pattern 3: Calculating Metrics

```python
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np

def calculate_metrics(y_true, y_pred):
    """Calculate regression metrics"""
    return {
        'r2': r2_score(y_true, y_pred),
        'mae': mean_absolute_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mape': np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    }

# Usage
for i, target in enumerate(['UCAOT', 'UCWOT', 'UCAF']):
    metrics = calculate_metrics(y_test[:, i], y_pred[:, i])
    print(f"\n{target}:")
    print(f"  R²: {metrics['r2']:.3f}")
    print(f"  MAE: {metrics['mae']:.3f}")
    print(f"  RMSE: {metrics['rmse']:.3f}")
    print(f"  MAPE: {metrics['mape']:.2f}%")
```

### Pattern 4: Progress Printing

```python
# Use this consistent format for script output
print("="*80)
print(" SCRIPT TITLE")
print("="*80)

print("\n[1/5] Loading data...")
# ... code ...
print("✓ Data loaded")

print("\n[2/5] Processing features...")
# ... code ...
print("✓ Features processed")

print("\n[3/5] Training models...")
# ... code ...
print("✓ Models trained")

print("\n[4/5] Validating...")
# ... code ...
print("✓ Validation complete")

print("\n[5/5] Saving results...")
# ... code ...
print("✓ Results saved")

print("\n" + "="*80)
print(" COMPLETED SUCCESSFULLY")
print("="*80)
```

### Pattern 5: Error Handling

```python
import sys
from pathlib import Path

# Check file existence
data_file = Path('data/processed_no_leakage/X_train_scaled.npy')
if not data_file.exists():
    print(f"ERROR: Data file not found: {data_file}")
    print("Please run: python run_sprint1_pipeline_no_leakage.py")
    sys.exit(1)

# Check directory existence
output_dir = Path('results')
output_dir.mkdir(exist_ok=True)

# Try-except for file operations
try:
    import joblib
    model = joblib.load('models/lightgbm_model_no_leakage.pkl')
except FileNotFoundError:
    print("ERROR: Model not found. Please train first.")
    sys.exit(1)
except Exception as e:
    print(f"ERROR loading model: {e}")
    sys.exit(1)
```

---

## 🔀 Git Workflow

### Branch Strategy

**Main Branch:** `main` (or default branch)
- Contains stable, production-ready code
- All features must pass validation before merging

**Feature Branches:**
- Format: `feature/<description>` (e.g., `feature/add-temporal-features`)
- Format: `fix/<description>` (e.g., `fix/fmu-scaling-issue`)
- Format: `sprint/<number>` (e.g., `sprint/6-deployment`)

**Claude Branches:**
- Auto-generated: `claude/<session-id>`
- Used for AI assistant work
- Must start with `claude/` and end with matching session ID

### Commit Message Format

```
<type>: <short description>

<optional longer description>

Examples:
- feat: Add temporal feature engineering
- fix: Correct FMU output descaling
- docs: Update CLAUDE.md with deployment guide
- refactor: Reorganize deployment scripts
- test: Add FMU validation tests
- chore: Update requirements.txt
```

### Pushing Changes

```bash
# ALWAYS push to correct branch
git push -u origin <branch-name>

# For Claude branches (CRITICAL):
# Branch must start with 'claude/' and end with session ID
# Otherwise push will fail with 403 error

# Retry logic for network issues:
# If push fails, retry up to 4 times with exponential backoff
# (2s, 4s, 8s, 16s)
```

### Typical Workflow

```bash
# 1. Check current status
git status

# 2. Stage changes
git add <files>

# 3. Commit with descriptive message
git commit -m "feat: Add new thermodynamic feature"

# 4. Push to remote
git push -u origin <branch-name>

# 5. Create pull request (via GitHub UI)
```

---

## 🐛 Troubleshooting

### Issue 1: Data Pipeline Fails

**Symptoms:** `run_sprint1_pipeline_no_leakage.py` crashes

**Causes:**
1. Raw data file missing in `data/raw/`
2. Incorrect CSV filename in `data_loader.py`
3. Missing dependencies

**Solutions:**
```bash
# Check raw data exists
ls -lh data/raw/

# If missing, download or copy data
# Expected: datos_combinados_entrenamiento_20251118_105234.csv

# Check dependencies
pip install -r requirements.txt

# Run with verbose output
python run_sprint1_pipeline_no_leakage.py
```

### Issue 2: Model Training Poor Performance

**Symptoms:** R² < 0.70 on validation set

**Possible Causes:**
1. Data leakage introduced (CHECK THIS FIRST!)
2. Data quality issues
3. Scaling issues
4. Temporal split issues

**Debug Steps:**
```bash
# 1. Verify no data leakage
python scripts/analysis/investigate_validation_data.py

# 2. Check feature engineering
# Review: src/data/feature_engineering_no_leakage.py
# Ensure NO usage of UCAOT, UCWOT, UCAF

# 3. Analyze data quality
python scripts/analysis/analyze_test_data_detail.py

# 4. Check scaling
# Load scalers and verify they're fit on training data only
```

### Issue 3: FMU Validation Fails

**Symptoms:** `validate_fmu_predictions.py` shows R² < 0.70

**Possible Causes:**
1. FMU not rebuilt after model changes
2. Scaler mismatch
3. Feature engineering mismatch
4. Model file corrupted

**Solutions:**
```bash
# 1. Clean rebuild
python deployment/scripts/clean_model_for_fmu.py
python deployment/scripts/clean_scaler_for_fmu.py
python deployment/scripts/export_fmu_sensor_inputs.py

# 2. Validate again
python deployment/validation/validate_fmu_predictions.py

# 3. Check FMU file size (should be ~2.4 MB)
ls -lh deployment/fmu/HVACUnitCoolerFMU.fmu

# 4. Quick functionality test
python deployment/validation/test_fmu.py
```

### Issue 4: FMU Simulation Error

**Symptoms:** FMPy simulation crashes or returns NaN

**Possible Causes:**
1. Input values out of training range
2. Missing sensor values
3. Invalid sensor values (negative, NaN)
4. FMU file corrupted

**Debug:**
```python
# Check input ranges (from training data)
import numpy as np

X_train = np.load('data/processed_no_leakage/X_train_scaled.npy')
scaler = joblib.load('data/processed_no_leakage/scaler.pkl')

# Unscale to get physical ranges
X_train_physical = scaler.inverse_transform(X_train)

# Print ranges for each sensor
with open('data/processed_no_leakage/metadata.json') as f:
    metadata = json.load(f)

for i, feature in enumerate(metadata['feature_names'][:20]):
    print(f"{feature}: [{X_train_physical[:, i].min():.2f}, "
          f"{X_train_physical[:, i].max():.2f}]")

# Compare your inputs to these ranges
# If out of range, extrapolation may fail
```

### Issue 5: Import Errors

**Symptoms:** `ModuleNotFoundError` or `ImportError`

**Solutions:**
```bash
# 1. Install all dependencies
pip install -r requirements.txt

# 2. For FMU-specific dependencies
pip install fmpy pythonfmu

# 3. For edge deployment
pip install -r requirements.edge.txt

# 4. For integration
pip install -r requirements.integration.txt

# 5. Verify Python version
python --version  # Should be 3.11+

# 6. Check virtual environment
which python
# Should point to venv/bin/python if using venv
```

### Issue 6: Git Push Fails (403)

**Symptoms:** `git push` returns 403 Forbidden

**Cause:** Branch name doesn't match required format for Claude branches

**Solution:**
```bash
# For Claude branches:
# Branch MUST start with 'claude/' and end with session ID

# Check current branch
git branch

# If incorrect, rename branch
git branch -m <old-name> claude/<session-id>

# Then push
git push -u origin claude/<session-id>
```

---

## 📚 Additional Resources

### Key Documentation Files

1. **README.md** - Main project documentation
2. **CHANGELOG_NO_LEAKAGE.md** - Development history
3. **deployment/README.md** - Deployment guide
4. **deployment/fmu/README_SENSOR_INPUTS.md** - FMU usage
5. **deployment/fmu/FMU_SETUP_GUIDE.md** - FMU setup
6. **scripts/README.md** - Utility scripts guide
7. **data/DATA_SUMMARY.md** - Dataset documentation

### Sprint Documentation

Located in `docs/`:
- Sprint0_Setup_Data_Exploration.md
- Sprint1_Data_Analysis_Preprocessing.md
- Sprint2_Model_Development_LightGBM.md
- Sprint3_PINN_Comprehensive_Analysis.md
- Sprint5_Comprehensive_Evaluation_Report.md
- Sprint6_Edge_Deployment.md
- Sprint7_RealTime_Integration.md
- Sprint8_Documentation_Knowledge_Transfer.md

### External References

- **LightGBM Docs:** https://lightgbm.readthedocs.io/
- **FMI Standard:** https://fmi-standard.org/
- **FMPy Docs:** https://github.com/CATIA-Systems/FMPy
- **PythonFMU:** https://github.com/NTNU-IHB/PythonFMU

---

## ✅ Quick Reference Checklist

### Starting a New Task

- [ ] Read README.md for context
- [ ] Review this CLAUDE.md for conventions
- [ ] Check current branch: `git status`
- [ ] Verify data exists: `ls data/processed_no_leakage/`
- [ ] Check model exists: `ls models/lightgbm_model_no_leakage.pkl`

### Adding Features

- [ ] Use ONLY sensor inputs (no targets!)
- [ ] Update `feature_engineering_no_leakage.py`
- [ ] Update feature_names in metadata
- [ ] Rerun data pipeline
- [ ] Retrain model
- [ ] Rebuild FMU
- [ ] Validate FMU

### Before Committing

- [ ] Code follows conventions
- [ ] No data leakage introduced
- [ ] Tests pass (if applicable)
- [ ] FMU validation passes (R² > 0.75)
- [ ] Documentation updated
- [ ] Commit message is descriptive

### Before Pushing

- [ ] Check branch name (claude/* for AI work)
- [ ] Stage all changes: `git add <files>`
- [ ] Commit: `git commit -m "type: description"`
- [ ] Push: `git push -u origin <branch-name>`

---

## 🎓 Learning Path for New AI Assistants

### Day 1: Understanding the Project
1. Read README.md (project overview)
2. Read CHANGELOG_NO_LEAKAGE.md (history)
3. Understand the NO DATA LEAKAGE principle
4. Review repository structure

### Day 2: Data Pipeline
1. Study `src/data/feature_engineering_no_leakage.py`
2. Understand temporal split strategy
3. Review scaling approach
4. Run: `python run_sprint1_pipeline_no_leakage.py`

### Day 3: Model Training
1. Review `train_model_no_leakage.py`
2. Understand LightGBM multi-output approach
3. Run: `python train_model_no_leakage.py`
4. Analyze validation metrics

### Day 4: FMU Deployment
1. Read `deployment/fmu/README_SENSOR_INPUTS.md`
2. Understand FMU internal process
3. Run: `python deployment/scripts/export_fmu_sensor_inputs.py`
4. Run: `python deployment/validation/validate_fmu_predictions.py`

### Day 5: Practice
1. Make a small feature addition
2. Retrain model
3. Rebuild FMU
4. Validate end-to-end
5. Commit changes following conventions

---

## 📞 Getting Help

When stuck, check in this order:

1. **This file (CLAUDE.md)** - Comprehensive guide
2. **README.md** - Project overview
3. **Specific READMEs** - deployment/README.md, scripts/README.md
4. **CHANGELOG** - Recent changes and decisions
5. **Sprint docs** - Deep dives into specific topics
6. **Code comments** - Implementation details

---

**Remember:**

- **NO DATA LEAKAGE** is the #1 rule
- Always use `_no_leakage` variants
- Validate FMU after changes (R² > 0.75)
- Follow naming conventions
- Document your changes
- Test thoroughly before committing

**Good luck! 🚀**
