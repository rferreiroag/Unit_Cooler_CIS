# HVACUnitCoolerFMU - Setup Guide

## Overview

This FMU (Functional Mock-up Unit) implements an HVAC Unit Cooler predictive model using LightGBM machine learning.

**Model Specifications:**
- **Inputs**: 20 sensor variables
- **Outputs**: 3 predictions (UCAOT, UCWOT, UCAF)
- **Performance**: R² = 0.80 (realistic for production)
- **Features**: 39 total (20 sensors + 19 computed internally)
- **Data Leakage**: NONE - all features computable in real-time

## System Requirements

### Required Software

1. **Python 3.7 or higher**
   - Download from: https://www.python.org/downloads/
   - **IMPORTANT for Windows**: Check "Add Python to PATH" during installation

2. **Python Packages**
   ```bash
   pip install numpy scikit-learn lightgbm joblib
   ```

3. **FMU Simulation Tool**
   - FMPy (Python)
   - Dymola/Modelica
   - MATLAB/Simulink
   - OpenModelica
   - Or any FMI 2.0 compatible tool

## Installation Instructions

### Windows

```cmd
REM 1. Install Python 3.7+ from python.org

REM 2. Open Command Prompt and install packages
python -m pip install --upgrade pip
python -m pip install numpy scikit-learn lightgbm joblib

REM 3. Verify installation
python -c "import numpy, sklearn, lightgbm, joblib; print('All packages installed successfully')"
```

### Linux/Mac

```bash
# 1. Install Python 3.7+ (usually pre-installed)

# 2. Install packages
pip3 install numpy scikit-learn lightgbm joblib

# 3. Verify installation
python3 -c "import numpy, sklearn, lightgbm, joblib; print('All packages installed successfully')"
```

## Testing the FMU

### Option 1: Test with FMPy (Recommended)

```bash
# Install FMPy
pip install fmpy

# Test the FMU
fmpy simulate HVACUnitCoolerFMU.fmu \
  --start-values AMBT=23.0 UCTSP=21.0 UCAIT=22.0 UCWIT=23.0 UCWF=100.0 \
  --stop-time 10.0 \
  --output-interval 1.0
```

### Option 2: Test with Python Script

```python
from fmpy import simulate_fmu

# Define sensor inputs
start_values = {
    'AMBT': 23.0,      # Ambient temperature
    'UCTSP': 21.0,     # Setpoint
    'UCAIT': 22.0,     # Air inlet temperature
    'UCWIT': 23.0,     # Water inlet temperature
    'UCWF': 100.0,     # Water flow
    # ... (add all 20 sensor inputs)
}

# Simulate
result = simulate_fmu(
    'HVACUnitCoolerFMU.fmu',
    start_values=start_values,
    stop_time=10.0
)

# Check outputs
print("UCAOT (Air Outlet Temperature):", result['UCAOT'][-1])
print("UCWOT (Water Outlet Temperature):", result['UCWOT'][-1])
print("UCAF (Air Flow):", result['UCAF'][-1])
```

## Input Variables (20 Sensors)

| Name | Description | Unit | Default |
|------|-------------|------|---------|
| AMBT | Ambient Temperature | °C | 20.0 |
| UCTSP | Temperature Setpoint | °C | 21.0 |
| CPSP | Compressor Setpoint | - | 0.0 |
| UCAIT | Air Inlet Temperature | °C | 20.0 |
| CPPR | Compressor Pressure | bar | 0.0 |
| UCWF | Water Flow | L/min | 0.0 |
| CPMC | Compressor Current | A | 0.0 |
| MVDP | Valve Differential Pressure | bar | 0.0 |
| CPCF | Compressor Frequency | Hz | 0.0 |
| UCFS | Fan Speed | RPM | 0.0 |
| MVCV | Valve Control Voltage | V | 0.0 |
| UCHV | Heater Voltage | V | 0.0 |
| CPMV | Compressor Motor Voltage | V | 0.0 |
| UCHC | Heater Current | A | 0.0 |
| UCWIT | Water Inlet Temperature | °C | 20.0 |
| UCFMS | Fan Motor Speed | RPM | 0.0 |
| CPDP | Compressor Discharge Pressure | bar | 0.0 |
| UCWDP | Water Differential Pressure | bar | 0.0 |
| MVWF | Valve Water Flow | L/min | 0.0 |
| UCOM | Operating Mode | - | 0.0 |

## Output Variables (3 Predictions)

| Name | Description | Unit | Range |
|------|-------------|------|-------|
| UCAOT | Air Outlet Temperature | °C | 15-30 |
| UCWOT | Water Outlet Temperature | °C | 15-30 |
| UCAF | Air Flow | m³/h | 500-1500 |

## Troubleshooting

### Error: "Could not load models"

**Cause**: Missing Python packages or incorrect Python version

**Solution**:
```bash
# Check Python version (must be 3.7+)
python --version

# Reinstall packages
pip install --force-reinstall numpy scikit-learn lightgbm joblib
```

### Error: "UnicodeEncodeError" or "charmap codec" (Windows)

**Cause**: Windows console encoding issue

**Solution**: This has been fixed in the latest FMU version. If you still see this:
```cmd
# Set environment variable before running
set PYTHONIOENCODING=utf-8
```

### Error: "No module named 'data.data_splits'"

**Cause**: Using old version of model files

**Solution**: This has been fixed. Make sure you're using the latest FMU file.

### FMU Returns Default Values (20.0, 20.0, 1000.0)

**Cause**: Models failed to load

**Solution**:
1. Check that Python and packages are installed correctly
2. Look at console output for error messages
3. Verify file permissions on FMU file

## Performance Metrics

The model has been trained on 56,211 samples with the following performance:

| Target | R² Score | MAE | RMSE | Notes |
|--------|----------|-----|------|-------|
| UCAOT | 0.9132 | 0.1359 | 0.2238 | Excellent |
| UCWOT | 0.7469 | 0.2531 | 0.5151 | Good |
| UCAF | 0.7541 | 0.2002 | 0.4719 | Good |
| **Average** | **0.8047** | - | - | Production-ready |

**Note**: These metrics are for normalized/scaled values. The model has NO data leakage and all features are computable from sensor inputs only.

## Integration with Simulation Tools

### Dymola/Modelica

1. Open Dymola
2. File → Import → FMU
3. Select `HVACUnitCoolerFMU.fmu`
4. Connect input sensors and read outputs

### MATLAB/Simulink

1. Add "FMU Import" block from FMI Kit
2. Select `HVACUnitCoolerFMU.fmu`
3. Configure inputs/outputs
4. Run simulation

### OpenModelica

```modelica
model TestHVAC
  FMI.FMU_Import.HVACUnitCoolerFMU fmu;
equation
  fmu.AMBT = 23.0;
  fmu.UCTSP = 21.0;
  // ... set other inputs
end TestHVAC;
```

## Support

For issues or questions:
1. Check this guide's Troubleshooting section
2. Verify all system requirements are met
3. Check console output for detailed error messages
4. Contact the development team with error logs

## Version Information

- **FMI Version**: 2.0 (Co-Simulation)
- **Model Version**: v1.0
- **Last Updated**: 2025-11-20
- **Tool**: PythonFMU 0.6.9
