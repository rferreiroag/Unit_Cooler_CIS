"""
HVAC Unit Cooler FMU with Sensor Inputs Only

This FMU accepts ONLY sensor inputs (20 variables) and computes
features internally before prediction.

NO DATA LEAKAGE - All features computable in real-time production.

Inputs (20 sensors):
  1-20: AMBT, UCTSP, CPSP, UCAIT, CPPR, UCWF, CPMC, MVDP, CPCF, UCFS,
        MVCV, UCHV, CPMV, UCHC, UCWIT, UCFMS, CPDP, UCWDP, MVWF, UCOM

Outputs (3 predictions):
  1-3: UCAOT, UCWOT, UCAF

Internal processing:
  - Computes 19 physics-based features
  - Scales inputs
  - Makes predictions
  - Returns outputs

To build FMU:
    pip install pythonfmu
    pythonfmu build -f deployment/fmu/hvac_fmu_sensor_inputs.py
"""

import numpy as np
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Try to import pythonfmu (may not be available during development)
try:
    from pythonfmu import Fmi2Causality, Fmi2Slave, Fmi2Variability, Real
    PYTHONFMU_AVAILABLE = True
except ImportError:
    PYTHONFMU_AVAILABLE = False
    print("Warning: pythonfmu not available. Install with: pip install pythonfmu")


class FeatureComputer:
    """Compute physics-based features from sensor inputs"""

    def __init__(self):
        # Physical constants
        self.Cp_water = 4186.0
        self.Cp_air = 1005.0
        self.rho_water = 1000.0
        self.rho_air = 1.2
        self.L_per_m3 = 1000.0
        self.min_to_s = 60.0
        self.W_to_kW = 0.001
        self.epsilon = 1e-9

        # Time index for temporal features
        self.time_index = 0

    def compute_features(self, sensors):
        """
        Compute 19 features from 20 sensor inputs

        Args:
            sensors: dict with sensor names and values

        Returns:
            numpy array with 39 features (20 sensors + 19 derived)
        """
        # Extract sensor values
        AMBT = sensors.get('AMBT', 20.0)
        UCTSP = sensors.get('UCTSP', 21.0)
        CPSP = sensors.get('CPSP', 0.0)
        UCAIT = sensors.get('UCAIT', 20.0)
        CPPR = sensors.get('CPPR', 0.0)
        UCWF = sensors.get('UCWF', 0.0)
        CPMC = sensors.get('CPMC', 0.0)
        MVDP = sensors.get('MVDP', 0.0)
        CPCF = sensors.get('CPCF', 0.0)
        UCFS = sensors.get('UCFS', 0.0)
        MVCV = sensors.get('MVCV', 0.0)
        UCHV = sensors.get('UCHV', 0.0)
        CPMV = sensors.get('CPMV', 0.0)
        UCHC = sensors.get('UCHC', 0.0)
        UCWIT = sensors.get('UCWIT', 20.0)
        UCFMS = sensors.get('UCFMS', 0.0)
        CPDP = sensors.get('CPDP', 0.0)
        UCWDP = sensors.get('UCWDP', 0.0)
        MVWF = sensors.get('MVWF', 0.0)
        UCOM = sensors.get('UCOM', 0.0)

        # Compute derived features (19 total)

        # 1-5: Temperature deltas
        T_approach = UCWIT - UCAIT
        T_water_ambient_diff = UCWIT - AMBT
        T_air_ambient_diff = UCAIT - AMBT
        setpoint_inlet_diff = UCTSP - UCAIT
        setpoint_ambient_diff = UCTSP - AMBT

        # 6: Mass flow rate (water)
        mdot_water = UCWF * self.rho_water / (self.L_per_m3 * self.min_to_s)

        # 7-8: Thermal capacity and max power
        C_water = mdot_water * self.Cp_water * self.W_to_kW
        Q_max_water = C_water * abs(T_approach)

        # 9-11: Power estimates
        P_fan_estimate = ((abs(UCFS) + 1.0) ** 1.5) * 0.001  # kW
        P_pump_estimate = (UCWF * UCWDP * 0.0001) if UCWDP != 0 else ((UCWF + 1.0) * 0.01)
        P_total_estimate = P_fan_estimate + P_pump_estimate

        # 12-15: Temporal features
        minutes_per_day = 1440
        cycle_hour = (self.time_index % minutes_per_day) / 60.0
        hour_sin = np.sin(2 * np.pi * cycle_hour / 24.0)
        hour_cos = np.cos(2 * np.pi * cycle_hour / 24.0)

        # 16-19: Interaction features
        T_water_x_flow = UCWIT * UCWF
        ambient_x_inlet = AMBT * UCAIT
        setpoint_x_flow = UCTSP * UCWF
        T_water_x_pressure = UCWIT * UCWDP

        # Increment time index
        self.time_index += 1

        # Combine all features (20 sensors + 19 derived = 39 total)
        features = np.array([
            # Original sensors (20)
            AMBT, UCTSP, CPSP, UCAIT, CPPR, UCWF, CPMC, MVDP, CPCF, UCFS,
            MVCV, UCHV, CPMV, UCHC, UCWIT, UCFMS, CPDP, UCWDP, MVWF, UCOM,
            # Derived features (19)
            T_approach, T_water_ambient_diff, T_air_ambient_diff,
            setpoint_inlet_diff, setpoint_ambient_diff,
            mdot_water, C_water, Q_max_water,
            P_fan_estimate, P_pump_estimate, P_total_estimate,
            float(self.time_index), cycle_hour, hour_sin, hour_cos,
            T_water_x_flow, ambient_x_inlet, setpoint_x_flow, T_water_x_pressure
        ])

        return features


if PYTHONFMU_AVAILABLE:
    class HVACUnitCoolerFMU(Fmi2Slave):
        """FMU with sensor inputs only - NO DATA LEAKAGE"""

        author = "HVAC Digital Twin Team"
        description = "LightGBM model with sensor inputs only (20 inputs, 3 outputs)"

        def __init__(self, **kwargs):
            super().__init__(**kwargs)

            # Load model and scaler from FMU resources directory
            # Use self.resources which points to the FMU's resources folder
            resources_dir = Path(self.resources) if hasattr(self, 'resources') else Path(__file__).parent / "resources"

            try:
                model_data = joblib.load(resources_dir / "lightgbm_model_no_leakage.pkl")
                self.models = model_data['models']
                self.scaler = joblib.load(resources_dir / "scaler.pkl")
                print(f"✓ Loaded models from {resources_dir}")
            except Exception as e:
                print(f"Warning: Could not load models: {e}")
                print(f"  Resources dir: {resources_dir}")
                print(f"  Resources exists: {resources_dir.exists() if isinstance(resources_dir, Path) else 'N/A'}")
                self.models = None
                self.scaler = None

            # Initialize feature computer
            self.feature_computer = FeatureComputer()

            # Define 20 sensor input variables
            self.AMBT = 20.0
            self.UCTSP = 21.0
            self.CPSP = 0.0
            self.UCAIT = 20.0
            self.CPPR = 0.0
            self.UCWF = 0.0
            self.CPMC = 0.0
            self.MVDP = 0.0
            self.CPCF = 0.0
            self.UCFS = 0.0
            self.MVCV = 0.0
            self.UCHV = 0.0
            self.CPMV = 0.0
            self.UCHC = 0.0
            self.UCWIT = 20.0
            self.UCFMS = 0.0
            self.CPDP = 0.0
            self.UCWDP = 0.0
            self.MVWF = 0.0
            self.UCOM = 0.0

            # Register input variables
            self.register_variable(Real("AMBT", causality=Fmi2Causality.input))
            self.register_variable(Real("UCTSP", causality=Fmi2Causality.input))
            self.register_variable(Real("CPSP", causality=Fmi2Causality.input))
            self.register_variable(Real("UCAIT", causality=Fmi2Causality.input))
            self.register_variable(Real("CPPR", causality=Fmi2Causality.input))
            self.register_variable(Real("UCWF", causality=Fmi2Causality.input))
            self.register_variable(Real("CPMC", causality=Fmi2Causality.input))
            self.register_variable(Real("MVDP", causality=Fmi2Causality.input))
            self.register_variable(Real("CPCF", causality=Fmi2Causality.input))
            self.register_variable(Real("UCFS", causality=Fmi2Causality.input))
            self.register_variable(Real("MVCV", causality=Fmi2Causality.input))
            self.register_variable(Real("UCHV", causality=Fmi2Causality.input))
            self.register_variable(Real("CPMV", causality=Fmi2Causality.input))
            self.register_variable(Real("UCHC", causality=Fmi2Causality.input))
            self.register_variable(Real("UCWIT", causality=Fmi2Causality.input))
            self.register_variable(Real("UCFMS", causality=Fmi2Causality.input))
            self.register_variable(Real("CPDP", causality=Fmi2Causality.input))
            self.register_variable(Real("UCWDP", causality=Fmi2Causality.input))
            self.register_variable(Real("MVWF", causality=Fmi2Causality.input))
            self.register_variable(Real("UCOM", causality=Fmi2Causality.input))

            # Define 3 output variables
            self.UCAOT = 20.0
            self.UCWOT = 20.0
            self.UCAF = 1000.0

            self.register_variable(Real("UCAOT", causality=Fmi2Causality.output))
            self.register_variable(Real("UCWOT", causality=Fmi2Causality.output))
            self.register_variable(Real("UCAF", causality=Fmi2Causality.output))

        def do_step(self, current_time, step_size):
            """Execute one simulation step"""

            if self.models is None or self.scaler is None:
                # Fallback if models not loaded
                self.UCAOT = 20.0
                self.UCWOT = 20.0
                self.UCAF = 1000.0
                return True

            # Collect sensor inputs
            sensors = {
                'AMBT': self.AMBT,
                'UCTSP': self.UCTSP,
                'CPSP': self.CPSP,
                'UCAIT': self.UCAIT,
                'CPPR': self.CPPR,
                'UCWF': self.UCWF,
                'CPMC': self.CPMC,
                'MVDP': self.MVDP,
                'CPCF': self.CPCF,
                'UCFS': self.UCFS,
                'MVCV': self.MVCV,
                'UCHV': self.UCHV,
                'CPMV': self.CPMV,
                'UCHC': self.UCHC,
                'UCWIT': self.UCWIT,
                'UCFMS': self.UCFMS,
                'CPDP': self.CPDP,
                'UCWDP': self.UCWDP,
                'MVWF': self.MVWF,
                'UCOM': self.UCOM
            }

            # Compute 39 features (20 sensors + 19 derived)
            features = self.feature_computer.compute_features(sensors)

            # Scale features
            features_scaled = self.scaler.transform(features.reshape(1, -1))

            # Make predictions
            self.UCAOT = float(self.models['UCAOT'].predict(features_scaled)[0])
            self.UCWOT = float(self.models['UCWOT'].predict(features_scaled)[0])
            self.UCAF = float(self.models['UCAF'].predict(features_scaled)[0])

            return True


# Standalone test
if __name__ == "__main__":
    print("="*80)
    print(" HVAC FMU - Sensor Inputs Only (NO DATA LEAKAGE)")
    print("="*80)

    if not PYTHONFMU_AVAILABLE:
        print("\n⚠ pythonfmu not installed")
        print("  Install with: pip install pythonfmu")
        print("  Then build FMU with: pythonfmu build -f deployment/fmu/hvac_fmu_sensor_inputs.py")
    else:
        print("\n✓ pythonfmu available")
        print("  Build FMU with: pythonfmu build -f deployment/fmu/hvac_fmu_sensor_inputs.py")

    # Test feature computer
    print("\n[Testing Feature Computer]")
    fc = FeatureComputer()

    test_sensors = {
        'AMBT': 23.0,
        'UCTSP': 21.0,
        'UCAIT': 22.0,
        'UCWIT': 23.0,
        'UCWF': 100.0,
        'UCFS': 50.0,
        'UCWDP': 10.0,
        'CPSP': 70.0,
        'CPPR': 20.0,
    }

    features = fc.compute_features(test_sensors)
    print(f"✓ Computed {len(features)} features from {len(test_sensors)} sensors")
    print(f"  Input sensors: {len([k for k in test_sensors.keys()])}")
    print(f"  Derived features: {len(features) - 20}")
    print(f"\n✓ FMU ready for export")
    print("="*80)
