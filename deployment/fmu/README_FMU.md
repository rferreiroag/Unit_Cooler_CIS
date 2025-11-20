
# HVAC Unit Cooler FMU - Manual Build Instructions

This directory contains the resources needed to create FMU (Functional Mock-up Unit)
files for the HVAC Unit Cooler Digital Twin models.

## FMU Contents

- modelDescription.xml: FMI 2.0 interface description
- resources/: Model files (LightGBM models, scaler, metadata)
- binaries/: Platform-specific shared libraries (need to be compiled)

## Building Full FMUs

To create complete FMU files with compiled binaries, use one of these tools:

### Option 1: PythonFMU (Recommended)

```bash
pip install pythonfmu

# This will create a proper FMU with Python runtime
pythonfmu build -f deployment/fmu/fmu_model.py
```

### Option 2: FMPy

```bash
pip install fmpy

# Use FMPy to validate and simulate the FMU
python -m fmpy.simulate deployment/fmu/hvac_unit_cooler.fmu --show-plot
```

### Option 3: Manual Compilation

1. Implement FMI 2.0 C interface in binaries/linux64/hvac_unit_cooler.so
2. Implement FMI 2.0 C interface in binaries/win64/hvac_unit_cooler.dll
3. Zip the entire directory structure as .fmu file

## Current Status

The models and metadata are ready. To create a fully functional FMU:

1. Install PythonFMU: `pip install pythonfmu`
2. Use the provided Python wrapper
3. Build with: `pythonfmu build`

## Model Performance

- Inputs: 52 features
- Outputs: 3 predictions
- Inference time: <1 ms
- Accuracy: R²=0.993-1.0

## Integration

Use the generated .fmu file with:
- Modelica/Dymola
- MATLAB/Simulink (requires Simulink FMU Import)
- OpenModelica
- JModelica
- Any FMI 2.0 compatible tool

## Feature Names (Inputs)

 1. AMBT
 2. UCTSP
 3. CPSP
 4. UCAIT
 5. CPPR
 6. UCWF
 7. CPMC
 8. MVDP
 9. CPCF
10. UCFS
... (42 more features)

## Target Names (Outputs)

1. UCAOT
2. UCWOT
3. UCAF
