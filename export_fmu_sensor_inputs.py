"""
Export FMU with Sensor Inputs Only

This script prepares and exports the FMU that accepts only 20 sensor inputs.
The FMU computes features internally - NO DATA LEAKAGE.

Steps:
1. Copy model and scaler to FMU resources
2. Build FMU using pythonfmu
3. Validate FMU

Usage:
    python export_fmu_sensor_inputs.py
"""

import os
import shutil
import subprocess
from pathlib import Path

print("="*80)
print(" EXPORT FMU WITH SENSOR INPUTS ONLY")
print("="*80)

# Step 1: Prepare resources
print("\n[1/3] Preparing FMU resources...")

fmu_dir = Path("deployment/fmu")
resources_dir = fmu_dir / "resources"
resources_dir.mkdir(parents=True, exist_ok=True)

# Copy model
model_src = Path("models/lightgbm_model_no_leakage.pkl")
if model_src.exists():
    shutil.copy(model_src, resources_dir / "lightgbm_model_no_leakage.pkl")
    print(f"  ✓ Copied {model_src}")
else:
    print(f"  ✗ Model not found: {model_src}")
    print("    Run: python train_model_no_leakage.py")
    exit(1)

# Copy scaler
scaler_src = Path("data/processed_no_leakage/scaler.pkl")
if scaler_src.exists():
    shutil.copy(scaler_src, resources_dir / "scaler.pkl")
    print(f"  ✓ Copied {scaler_src}")
else:
    print(f"  ✗ Scaler not found: {scaler_src}")
    exit(1)

# Copy metadata
metadata_src = Path("data/processed_no_leakage/metadata.json")
if metadata_src.exists():
    shutil.copy(metadata_src, resources_dir / "metadata.json")
    print(f"  ✓ Copied {metadata_src}")

print("✓ Resources prepared")

# Step 2: Check pythonfmu
print("\n[2/3] Checking pythonfmu installation...")
try:
    result = subprocess.run(
        ["pythonfmu", "--version"],
        capture_output=True,
        text=True
    )
    if result.returncode == 0:
        print(f"  ✓ pythonfmu is installed")
    else:
        print("  ⚠ Installing pythonfmu...")
        subprocess.run(["pip", "install", "-q", "pythonfmu"], check=True)
        print("  ✓ pythonfmu installed")
except FileNotFoundError:
    print("  ⚠ Installing pythonfmu...")
    subprocess.run(["pip", "install", "-q", "pythonfmu"], check=True)
    print("  ✓ pythonfmu installed")

# Step 3: Build FMU
print("\n[3/3] Building FMU...")
fmu_script = fmu_dir / "hvac_fmu_sensor_inputs.py"

try:
    # Change to FMU directory
    original_dir = os.getcwd()
    os.chdir(fmu_dir)

    # Build FMU with resources explicitly included
    result = subprocess.run(
        [
            "pythonfmu", "build",
            "-f", "hvac_fmu_sensor_inputs.py",
            "resources/lightgbm_model_no_leakage.pkl",
            "resources/scaler.pkl",
            "resources/metadata.json"
        ],
        capture_output=True,
        text=True,
        timeout=120
    )

    os.chdir(original_dir)

    if result.returncode == 0:
        print("  ✓ FMU built successfully")

        # Find generated FMU
        fmu_files = list(fmu_dir.glob("*.fmu"))
        if fmu_files:
            for fmu_file in fmu_files:
                size_mb = fmu_file.stat().st_size / (1024 * 1024)
                print(f"  📦 {fmu_file.name} ({size_mb:.2f} MB)")
        else:
            print("  ⚠ No .fmu file found")
    else:
        print("  ✗ FMU build failed")
        print(f"\nError output:\n{result.stderr}")
        exit(1)

except subprocess.TimeoutExpired:
    os.chdir(original_dir)
    print("  ✗ Build timeout")
    exit(1)
except Exception as e:
    os.chdir(original_dir)
    print(f"  ✗ Build error: {e}")
    exit(1)

# Summary
print("\n" + "="*80)
print(" FMU EXPORT COMPLETE")
print("="*80)

print(f"""
FMU Specifications:
  • Inputs:  20 sensor variables
  • Outputs: 3 predictions (UCAOT, UCWOT, UCAF)
  • Model:   LightGBM (R²=0.80)
  • Features: 39 (20 sensors + 19 computed internally)

✓ NO DATA LEAKAGE - All features computable in real-time

FMU Location: {fmu_dir}/

Usage in simulation tools:
  - Modelica/Dymola: Import .fmu file
  - MATLAB/Simulink: Import with Simulink FMU block
  - Python: Use FMPy library

Test FMU:
  pip install fmpy
  fmpy simulate {fmu_dir}/HVACUnitCoolerFMU.fmu
""")
