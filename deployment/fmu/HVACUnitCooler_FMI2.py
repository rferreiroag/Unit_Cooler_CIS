"""
HVAC Unit Cooler Digital Twin FMU - FMI 2.0

Python-based FMU for co-simulation with OpenModelica, JModelica, etc.

Author: HVAC Digital Twin Team
Version: 1.0.0
FMI Standard: 2.0 Co-Simulation

Inputs: 52 features (HVAC sensors + engineered features)
Outputs: 3 predictions (UCAOT, UCWOT, UCAF)
Performance: R²=0.993-1.0, MAPE=0.008-8.7%

To build:
    pip install pythonfmu
    pythonfmu build -f HVACUnitCooler_FMI2.py

To simulate:
    pip install fmpy
    fmpy simulate HVACUnitCooler.fmu --show-plot
"""

from pythonfmu import Fmi2Causality, Fmi2Slave, Fmi2Variability, Real
from pathlib import Path
import numpy as np


class HVACUnitCooler(Fmi2Slave):
    """
    FMU for HVAC Unit Cooler Digital Twin using LightGBM models.

    Predicts:
    - UCAOT: Unit Cooler Air Outlet Temperature (°C)
    - UCWOT: Unit Cooler Water Outlet Temperature (°C)
    - UCAF: Unit Cooler Air Flow

    Based on 52 input features from HVAC sensors and physics-based
    engineered features.
    """

    author = "HVAC Digital Twin Team"
    description = "LightGBM ML Digital Twin (R²=0.993-1.0, <1ms inference)"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Initialize 52 input variables with default values
        # Raw Sensor Features (20)
        self.AMBT = 25.0
        self.UCTSP = 20.0
        self.CPSP = 10.0
        self.UCAIT = 25.0
        self.CPPR = 2.0
        self.UCWF = 1.0
        self.CPMC = 50.0
        self.MVDP = 0.5
        self.CPCF = 1.5
        self.UCFS = 1500.0
        self.MVCV = 0.5
        self.UCHV = 0.3
        self.CPMV = 0.5
        self.UCHC = 0.2
        self.UCWIT = 15.0
        self.UCFMS = 1500.0
        self.CPDP = 0.3
        self.UCWDP = 0.2
        self.MVWF = 1.0
        self.UCOM = 1.0

        # Temperature Features (5)
        self.delta_T_water = 4.0
        self.delta_T_air = 2.5
        self.T_approach = 5.0
        self.T_water_avg = 13.0
        self.T_air_avg = 23.0

        # Thermal Power Features (7)
        self.mdot_water = 1000.0
        self.mdot_air = 1.2
        self.Q_water = 16.7
        self.Q_air = 15.0
        self.Q_avg = 15.85
        self.Q_imbalance = 1.7
        self.Q_imbalance_pct = 10.0

        # Heat Exchanger Performance (4)
        self.efficiency_HX = 0.90
        self.effectiveness = 0.85
        self.NTU = 2.5
        self.C_ratio = 0.8

        # Fluid Dynamics (2)
        self.Re_air_estimate = 5000.0
        self.flow_ratio = 1.2

        # Control Features (3)
        self.delta_T_ratio = 0.625
        self.setpoint_error = 3.0
        self.setpoint_error_abs = 3.0

        # Power & Efficiency (4)
        self.P_fan_estimate = 0.5
        self.P_pump_estimate = 0.3
        self.P_total_estimate = 0.8
        self.COP_estimate = 19.8

        # Temporal Features (5)
        self.time_index = 0.0
        self.cycle_hour = 0.5
        self.hour_sin = 0.0
        self.hour_cos = 1.0

        # Interaction Features (3)
        self.T_water_x_flow = 13.0
        self.T_air_x_flow = 27.6
        self.ambient_x_inlet = 625.0

        # Output variables (3 predictions)
        self.UCAOT = 0.0
        self.UCWOT = 0.0
        self.UCAF = 0.0

        # Register all variables with FMI 2.0
        # Inputs (52 features)
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
        self.register_variable(Real("delta_T_water", causality=Fmi2Causality.input))
        self.register_variable(Real("delta_T_air", causality=Fmi2Causality.input))
        self.register_variable(Real("T_approach", causality=Fmi2Causality.input))
        self.register_variable(Real("T_water_avg", causality=Fmi2Causality.input))
        self.register_variable(Real("T_air_avg", causality=Fmi2Causality.input))
        self.register_variable(Real("mdot_water", causality=Fmi2Causality.input))
        self.register_variable(Real("mdot_air", causality=Fmi2Causality.input))
        self.register_variable(Real("Q_water", causality=Fmi2Causality.input))
        self.register_variable(Real("Q_air", causality=Fmi2Causality.input))
        self.register_variable(Real("Q_avg", causality=Fmi2Causality.input))
        self.register_variable(Real("Q_imbalance", causality=Fmi2Causality.input))
        self.register_variable(Real("Q_imbalance_pct", causality=Fmi2Causality.input))
        self.register_variable(Real("efficiency_HX", causality=Fmi2Causality.input))
        self.register_variable(Real("effectiveness", causality=Fmi2Causality.input))
        self.register_variable(Real("NTU", causality=Fmi2Causality.input))
        self.register_variable(Real("C_ratio", causality=Fmi2Causality.input))
        self.register_variable(Real("Re_air_estimate", causality=Fmi2Causality.input))
        self.register_variable(Real("flow_ratio", causality=Fmi2Causality.input))
        self.register_variable(Real("delta_T_ratio", causality=Fmi2Causality.input))
        self.register_variable(Real("setpoint_error", causality=Fmi2Causality.input))
        self.register_variable(Real("setpoint_error_abs", causality=Fmi2Causality.input))
        self.register_variable(Real("P_fan_estimate", causality=Fmi2Causality.input))
        self.register_variable(Real("P_pump_estimate", causality=Fmi2Causality.input))
        self.register_variable(Real("P_total_estimate", causality=Fmi2Causality.input))
        self.register_variable(Real("COP_estimate", causality=Fmi2Causality.input))
        self.register_variable(Real("time_index", causality=Fmi2Causality.input))
        self.register_variable(Real("cycle_hour", causality=Fmi2Causality.input))
        self.register_variable(Real("hour_sin", causality=Fmi2Causality.input))
        self.register_variable(Real("hour_cos", causality=Fmi2Causality.input))
        self.register_variable(Real("T_water_x_flow", causality=Fmi2Causality.input))
        self.register_variable(Real("T_air_x_flow", causality=Fmi2Causality.input))
        self.register_variable(Real("ambient_x_inlet", causality=Fmi2Causality.input))

        # Outputs (3 predictions)
        self.register_variable(Real("UCAOT", causality=Fmi2Causality.output))
        self.register_variable(Real("UCWOT", causality=Fmi2Causality.output))
        self.register_variable(Real("UCAF", causality=Fmi2Causality.output))

        # ML models (loaded in setup_experiment)
        self.models = None
        self.scaler = None

    def setup_experiment(self, start_time):
        """Load models from resources directory"""
        try:
            import joblib
            resources_dir = Path(self.resources)

            # Load LightGBM models
            model_path = resources_dir / "lightgbm_model.pkl"
            models_data = joblib.load(str(model_path))
            self.models = models_data['models']

            # Load scaler
            scaler_path = resources_dir / "scaler.pkl"
            self.scaler = joblib.load(str(scaler_path))

            return True
        except Exception as e:
            print(f"Error loading models: {e}")
            return False

    def do_step(self, current_time, step_size):
        """Execute one simulation step"""
        if self.models is None or self.scaler is None:
            return False

        try:
            # Collect all 52 inputs in correct order
            inputs = np.array([
                self.AMBT, self.UCTSP, self.CPSP, self.UCAIT, self.CPPR,
                self.UCWF, self.CPMC, self.MVDP, self.CPCF, self.UCFS,
                self.MVCV, self.UCHV, self.CPMV, self.UCHC, self.UCWIT,
                self.UCFMS, self.CPDP, self.UCWDP, self.MVWF, self.UCOM,
                self.delta_T_water, self.delta_T_air, self.T_approach,
                self.T_water_avg, self.T_air_avg, self.mdot_water, self.mdot_air,
                self.Q_water, self.Q_air, self.Q_avg, self.Q_imbalance,
                self.Q_imbalance_pct, self.efficiency_HX, self.effectiveness,
                self.NTU, self.C_ratio, self.Re_air_estimate, self.flow_ratio,
                self.delta_T_ratio, self.setpoint_error, self.setpoint_error_abs,
                self.P_fan_estimate, self.P_pump_estimate, self.P_total_estimate,
                self.COP_estimate, self.time_index, self.cycle_hour,
                self.hour_sin, self.hour_cos, self.T_water_x_flow,
                self.T_air_x_flow, self.ambient_x_inlet
            ])

            # Scale inputs
            X_scaled = self.scaler.transform(inputs.reshape(1, -1))

            # Make predictions
            self.UCAOT = float(self.models['UCAOT'].predict(X_scaled)[0])
            self.UCWOT = float(self.models['UCWOT'].predict(X_scaled)[0])
            self.UCAF = float(self.models['UCAF'].predict(X_scaled)[0])

            return True

        except Exception as e:
            print(f"Error in do_step: {e}")
            return False
