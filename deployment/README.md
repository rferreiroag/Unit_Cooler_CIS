# Deployment Directory

This directory contains all deployment-related files, scripts, and packages for the HVAC Unit Cooler Digital Twin.

## 📂 Directory Structure

```
deployment/
├── fmu/                    # FMI 2.0 Co-Simulation FMU
│   ├── HVACUnitCoolerFMU.fmu         # ⭐ Production FMU (2.4 MB)
│   ├── hvac_fmu_sensor_inputs.py     # FMU implementation
│   ├── resources/                     # FMU internal resources
│   ├── README_SENSOR_INPUTS.md        # FMU usage guide
│   ├── FMU_SETUP_GUIDE.md            # Setup instructions
│   └── REQUIREMENTS_FMU.txt           # FMU dependencies
│
├── scripts/                # FMU build and export scripts
│   ├── export_fmu_sensor_inputs.py   # Main FMU builder
│   ├── extract_y_scaler_for_fmu.py   # Extract output scaler
│   ├── clean_model_for_fmu.py        # Clean model artifacts
│   ├── clean_scaler_for_fmu.py       # Clean scaler artifacts
│   └── export_model_to_onnx.py       # ONNX export (optional)
│
├── validation/             # FMU testing and validation
│   ├── validate_fmu_predictions.py   # Full validation (100 samples)
│   ├── test_fmu.py                   # Basic FMU test
│   ├── test_fmu_comprehensive.py     # Comprehensive test
│   └── example_inference.py          # Usage examples
│
├── packages/               # Release packages
│   ├── test_data_package.zip         # Test data (121 KB)
│   ├── validation_data_package.zip   # Validation data (676 KB)
│   └── hvac_models_package.tar.gz    # Models package
│
├── onnx/                   # ONNX deployment (optional)
├── docker/                 # Docker containers (optional)
├── benchmarks/             # Performance benchmarking
└── README.md               # This file
```

## 🚀 Quick Start

### Build FMU

```bash
# Build production FMU
cd deployment/scripts
python export_fmu_sensor_inputs.py

# Output: deployment/fmu/HVACUnitCoolerFMU.fmu
```

### Validate FMU

```bash
# Run full validation
cd deployment/validation
python validate_fmu_predictions.py

# Expected output:
# UCAOT R²=0.92, UCWOT R²=0.76, UCAF R²=0.67
# Average R²=0.78
```

### Use FMU

```python
from fmpy import simulate_fmu

# Define sensor inputs (physical values)
sensors = {
    'UCWIT': 7.5,    # Water inlet temp (°C)
    'UCAIT': 25.0,   # Air inlet temp (°C)
    'UCWF': 15.0,    # Water flow (L/min)
    'AMBT': 22.0,    # Ambient temp (°C)
    'UCTSP': 21.0,   # Setpoint (°C)
    'UCWP': 2.5,     # Water pressure (bar)
    'CPDP': 8.5,     # Compressor discharge pressure (bar)
    'CPSP': 4.2,     # Compressor suction pressure (bar)
    # ... + 12 more sensors (see README_SENSOR_INPUTS.md)
}

# Simulate
result = simulate_fmu(
    'deployment/fmu/HVACUnitCoolerFMU.fmu',
    start_values=sensors,
    stop_time=1.0
)

# Get predictions
UCAOT = result['UCAOT'][-1]  # °C
UCWOT = result['UCWOT'][-1]  # °C
UCAF = result['UCAF'][-1]    # m³/h
```

## 📦 Packages

### 1. Test Data Package

**File:** `packages/test_data_package.zip` (121 KB)

Contains test data for validation and benchmarking:
- X_test_scaled.npy - Test features (8,432 × 39)
- y_test_scaled.npy - Test targets (8,432 × 3)
- scaler_clean.pkl - Input scaler
- y_scaler_clean.pkl - Output scaler
- metadata.json - Feature/target names

**Usage:**
```python
import numpy as np
import joblib

X_test = np.load('X_test_scaled.npy')
y_test = np.load('y_test_scaled.npy')
scaler_X = joblib.load('scaler_clean.pkl')
scaler_y = joblib.load('y_scaler_clean.pkl')

# Unscale to physical values
X_real = scaler_X.inverse_transform(X_test)
y_real = scaler_y.inverse_transform(y_test)
```

### 2. Validation Data Package

**File:** `packages/validation_data_package.zip` (676 KB)

Contains documentation and raw data for validation:
- investigate_validation_data.py - Data source analysis
- analyze_test_data_detail.py - Detailed test set analysis
- datos_combinados_entrenamiento_20251118_105234.csv - Raw data (6.5 MB)

### 3. Models Package

**File:** `packages/hvac_models_package.tar.gz`

Contains trained models and artifacts:
- lightgbm_model_no_leakage.pkl - Trained LightGBM models
- scaler.pkl - Input/output scalers
- metadata.json - Model metadata

## 🔧 FMU Specifications

**File:** `fmu/HVACUnitCoolerFMU.fmu`

| Specification | Value |
|---------------|-------|
| Standard | FMI 2.0 Co-Simulation |
| Size | 2.4 MB |
| Inputs | 20 sensors (physical units) |
| Outputs | 3 predictions (UCAOT, UCWOT, UCAF) |
| Internal Features | 39 (20 + 19 computed) |
| Model | LightGBM (R²=0.78-0.92) |
| Inference Time | <1 ms |
| Data Leakage | None ✅ |

### Inputs (20 sensors)

All inputs are in **physical units** (not scaled):

1. **UCWIT** - Water Inlet Temperature (°C)
2. **UCAIT** - Air Inlet Temperature (°C)
3. **UCWF** - Water Flow (L/min)
4. **AMBT** - Ambient Temperature (°C)
5. **UCTSP** - Temperature Setpoint (°C)
6. **UCWP** - Water Pressure (bar)
7. **CPDP** - Compressor Discharge Pressure (bar)
8. **CPSP** - Compressor Suction Pressure (bar)
9. **CPPR** - Compressor Power (W)
10. **UCFMS** - Fan Speed (RPM)
11. **UCFMV** - Fan Voltage (V)
12. **UCFS** - Fan Status (0/1)
13. **CPCF** - Compressor Flow (L/min)
14. **MVWF** - Mixing Valve Water Flow (L/min)
15. **CPMHP** - Compressor Motor HP (HP)
16. **CPMEP** - Compressor Motor Power (W)
17. **CPMMP** - Compressor Motor Max Power (W)
18. **CPHE** - Compressor Hours (h)
19. **CPOC** - Compressor On/Off Count
20. **CPES** - Compressor Error Status

### Outputs (3 predictions)

All outputs are in **physical units** (descaled internally):

1. **UCAOT** - Air Outlet Temperature (°C)
2. **UCWOT** - Water Outlet Temperature (°C)
3. **UCAF** - Air Flow (m³/h)

## 🛠️ Build Scripts

### export_fmu_sensor_inputs.py

Main FMU builder script.

**Usage:**
```bash
python export_fmu_sensor_inputs.py
```

**Process:**
1. Copies models and scalers to FMU resources
2. Validates FMU implementation
3. Builds FMU using pythonfmu
4. Generates HVACUnitCoolerFMU.fmu (2.4 MB)

### extract_y_scaler_for_fmu.py

Extracts output scaler from AdaptiveScaler wrapper.

**Usage:**
```bash
python extract_y_scaler_for_fmu.py
```

**Output:** `data/processed_no_leakage/y_scaler_clean.pkl`

### clean_model_for_fmu.py

Removes unnecessary model artifacts for smaller FMU size.

### clean_scaler_for_fmu.py

Creates clean scaler without module dependencies.

## ✅ Validation Scripts

### validate_fmu_predictions.py

Comprehensive FMU validation with 100 random test samples.

**Usage:**
```bash
python validate_fmu_predictions.py
```

**Output:**
- Prediction accuracy metrics (R², MAE, RMSE, MAPE)
- Sample predictions comparison
- Statistical analysis

**Expected Results:**
- UCAOT: R²=0.92, MAE=1.75°C
- UCWOT: R²=0.76, MAE=15.51°C
- UCAF: R²=0.67, MAE=340.86 m³/h
- Average: R²=0.78 ✅

### test_fmu.py

Quick FMU functionality test.

**Usage:**
```bash
python test_fmu.py
```

### test_fmu_comprehensive.py

Comprehensive FMU testing across operating conditions.

## 🔬 Technical Details

### FMU Internal Process

```
Sensor Inputs (physical)
    ↓
Input Scaling (StandardScaler)
    ↓
Feature Engineering (19 features)
    ↓
LightGBM Prediction (3 models)
    ↓
Output Descaling (StandardScaler⁻¹)
    ↓
Predictions (physical)
```

### No Data Leakage

All 39 features are computable from 20 sensor inputs:
- ✅ No dependency on target variables (UCAOT, UCWOT, UCAF)
- ✅ All features available in real-time
- ✅ Production-ready deployment

### Physics-Based Features (19)

Computed internally from sensors:
1. Temperature differences (5 features)
2. Thermodynamic properties (6 features)
3. Temporal patterns (4 features)
4. Interaction terms (4 features)

## 🌐 Integration

### Modelica/Dymola

```modelica
model HVACSystem
  HVACUnitCoolerFMU unitCooler;
equation
  // Connect sensors
  unitCooler.UCWIT = waterInletTemp;
  unitCooler.UCAIT = airInletTemp;
  // ... other sensors

  // Read predictions
  airOutletTemp = unitCooler.UCAOT;
  waterOutletTemp = unitCooler.UCWOT;
  airFlow = unitCooler.UCAF;
end HVACSystem;
```

### MATLAB/Simulink

1. Add FMU Import block from Simulink library
2. Select `HVACUnitCoolerFMU.fmu`
3. Connect sensor signals to inputs
4. Read predictions from outputs

### Python (FMPy)

```python
from fmpy import simulate_fmu

result = simulate_fmu(
    'HVACUnitCoolerFMU.fmu',
    start_values={'UCWIT': 7.5, ...},
    stop_time=1.0
)
```

## 📊 Performance

### Validation Results

| Metric | UCAOT | UCWOT | UCAF | Average |
|--------|-------|-------|------|---------|
| R² | 0.924 | 0.760 | 0.665 | **0.783** |
| MAE | 1.75°C | 15.51°C | 340.86 m³/h | - |
| RMSE | 2.42°C | 19.25°C | 884.33 m³/h | - |

### Inference Performance

- **Latency:** <1 ms
- **Throughput:** >1000 predictions/sec
- **Memory:** <10 MB runtime

## 🔄 Maintenance

### Rebuilding FMU

When to rebuild:
- Model retrained with new data
- Scaler parameters updated
- FMU implementation modified
- New physics features added

### Validation

Run validation after any changes:
```bash
python deployment/validation/validate_fmu_predictions.py
```

Expected: R² > 0.75 average

## 📚 Documentation

- **README_SENSOR_INPUTS.md** - Complete FMU usage guide
- **FMU_SETUP_GUIDE.md** - Setup and installation
- **REQUIREMENTS_FMU.txt** - FMU dependencies

## 🤝 Support

For issues or questions:
1. Check documentation in `fmu/` directory
2. Run validation scripts
3. Review example code in `validation/example_inference.py`
4. Contact: [rferreiroag](https://github.com/rferreiroag)

---

**Last Updated:** 2025-11-21
**FMU Version:** 1.0.0
**Status:** ✅ Production Ready
