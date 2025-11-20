"""
Test FMU to verify it loads models correctly
"""
try:
    from fmpy import simulate_fmu
    import numpy as np

    print("="*80)
    print(" Testing HVACUnitCoolerFMU.fmu")
    print("="*80)

    fmu_path = 'deployment/fmu/HVACUnitCoolerFMU.fmu'

    # Define test inputs (20 sensors)
    start_values = {
        'AMBT': 23.0,
        'UCTSP': 21.0,
        'CPSP': 70.0,
        'UCAIT': 22.0,
        'CPPR': 20.0,
        'UCWF': 100.0,
        'CPMC': 0.0,
        'MVDP': 10.0,
        'CPCF': 0.0,
        'UCFS': 50.0,
        'MVCV': 0.0,
        'UCHV': 0.0,
        'CPMV': 0.0,
        'UCHC': 0.0,
        'UCWIT': 23.0,
        'UCFMS': 0.0,
        'CPDP': 0.0,
        'UCWDP': 10.0,
        'MVWF': 0.0,
        'UCOM': 1.0
    }

    print("\n[Testing FMU with sample inputs]")
    print(f"FMU: {fmu_path}")
    print(f"Inputs: {len(start_values)} sensor variables")

    # Simulate
    result = simulate_fmu(
        fmu_path,
        start_values=start_values,
        stop_time=10.0,
        output_interval=1.0,
        fmi_call_logger=print  # Print all FMI calls for debugging
    )

    print("\n✓ Simulation completed")
    print(f"Result columns: {result.dtype.names}")

    # Check outputs
    print("\nOutput values:")
    print(f"  UCAOT: {result['UCAOT'][-1]:.2f}")
    print(f"  UCWOT: {result['UCWOT'][-1]:.2f}")
    print(f"  UCAF: {result['UCAF'][-1]:.2f}")

    print("\n" + "="*80)

except ImportError as e:
    print(f"⚠ Missing dependency: {e}")
    print("\nInstall with: pip install fmpy matplotlib")
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
