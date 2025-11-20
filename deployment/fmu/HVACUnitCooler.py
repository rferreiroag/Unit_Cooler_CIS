"""
HVAC Unit Cooler Digital Twin FMU

FMI 2.0 Co-Simulation implementation of LightGBM-based digital twin
for HVAC Unit Cooler prediction.

Author: HVAC Digital Twin Team
Version: 1.0.0
Date: 2025-11-20

Inputs: 52 features (sensor measurements and engineered features)
Outputs: 3 predictions (UCAOT, UCWOT, UCAF)

Performance:
- R²: 0.993-1.0
- MAPE: 0.008-8.7%
- Inference time: <1 ms

To build FMU:
    pip install pythonfmu3
    pythonfmu3 build HVACUnitCooler.py

To simulate:
    pip install fmpy
    fmpy simulate HVACUnitCooler.fmu --show-plot
"""

from pythonfmu3 import Fmi3Slave, Fmi3Causality, Fmi3Variability
from pathlib import Path
import numpy as np
import joblib
import json


class HVACUnitCooler(Fmi3Slave):
    """
    FMU for HVAC Unit Cooler Digital Twin using LightGBM models.

    This FMU predicts:
    - UCAOT: Unit Cooler Air Outlet Temperature (°C)
    - UCWOT: Unit Cooler Water Outlet Temperature (°C)
    - UCAF: Unit Cooler Air Flow

    Based on 52 input features including sensor measurements
    and physics-based engineered features.
    """

    author = "HVAC Digital Twin Team"
    description = "LightGBM ML model for HVAC unit cooler prediction (R²=0.993-1.0)"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # =================================================================
        # Input Variables (52 features)
        # =================================================================

        # Raw Sensor Features (20)
        self.AMBT = 25.0  # Ambient Temperature (°C)
        self.UCTSP = 20.0  # Unit Cooler Temperature Setpoint (°C)
        self.CPSP = 10.0  # Chiller Plant Setpoint (°C)
        self.UCAIT = 25.0  # Unit Cooler Air Inlet Temperature (°C)
        self.CPPR = 2.0  # Chiller Plant Pressure (bar)
        self.UCWF = 1.0  # Unit Cooler Water Flow (m³/s)
        self.CPMC = 50.0  # Chiller Plant Motor Current (A)
        self.MVDP = 0.5  # Mixing Valve Differential Pressure (bar)
        self.CPCF = 1.5  # Chiller Plant Cooling Flow (m³/s)
        self.UCFS = 1500.0  # Unit Cooler Fan Speed (RPM)
        self.MVCV = 0.5  # Mixing Valve Control Valve (0-1)
        self.UCHV = 0.3  # Unit Cooler Heating Valve (0-1)
        self.CPMV = 0.5  # Chiller Plant Mixing Valve (0-1)
        self.UCHC = 0.2  # Unit Cooler Heating Coil (0-1)
        self.UCWIT = 15.0  # Unit Cooler Water Inlet Temperature (°C)
        self.UCFMS = 1500.0  # Unit Cooler Fan Motor Speed (RPM)
        self.CPDP = 0.3  # Chiller Plant Differential Pressure (bar)
        self.UCWDP = 0.2  # Unit Cooler Water Differential Pressure (bar)
        self.MVWF = 1.0  # Mixing Valve Water Flow (m³/s)
        self.UCOM = 1.0  # Unit Cooler Operating Mode (enum)

        # Temperature Features (5)
        self.delta_T_water = 4.0  # Water temperature drop (°C)
        self.delta_T_air = 2.5  # Air temperature rise (°C)
        self.T_approach = 5.0  # Approach temperature (°C)
        self.T_water_avg = 13.0  # Average water temperature (°C)
        self.T_air_avg = 23.0  # Average air temperature (°C)

        # Thermal Power Features (6)
        self.mdot_water = 1000.0  # Water mass flow rate (kg/s)
        self.mdot_air = 1.2  # Air mass flow rate (kg/s)
        self.Q_water = 16.7  # Heat released by water (kW)
        self.Q_air = 15.0  # Heat absorbed by air (kW)
        self.Q_avg = 15.85  # Average thermal power (kW)
        self.Q_imbalance = 1.7  # Energy imbalance (kW)
        self.Q_imbalance_pct = 10.0  # Energy imbalance (%)

        # Heat Exchanger Performance (4)
        self.efficiency_HX = 0.90  # Heat exchanger efficiency (0-1)
        self.effectiveness = 0.85  # Heat exchanger effectiveness (0-1)
        self.NTU = 2.5  # Number of Transfer Units
        self.C_ratio = 0.8  # Heat capacity ratio

        # Fluid Dynamics Features (2)
        self.Re_air_estimate = 5000.0  # Reynolds number estimate
        self.flow_ratio = 1.2  # Air to water flow ratio

        # Control Features (3)
        self.delta_T_ratio = 0.625  # Temperature delta ratio
        self.setpoint_error = 3.0  # Setpoint tracking error (°C)
        self.setpoint_error_abs = 3.0  # Absolute setpoint error (°C)

        # Power & Efficiency Features (4)
        self.P_fan_estimate = 0.5  # Fan power estimate (kW)
        self.P_pump_estimate = 0.3  # Pump power estimate (kW)
        self.P_total_estimate = 0.8  # Total power estimate (kW)
        self.COP_estimate = 19.8  # Coefficient of Performance

        # Temporal Features (5)
        self.time_index = 0.0  # Time index for sequential patterns
        self.cycle_hour = 0.5  # Operational cycle hour (0-1)
        self.hour_sin = 0.0  # Sine component of hour
        self.hour_cos = 1.0  # Cosine component of hour

        # Interaction Features (3)
        self.T_water_x_flow = 13.0  # Temperature-flow interaction
        self.T_air_x_flow = 27.6  # Air temperature-flow interaction
        self.ambient_x_inlet = 625.0  # Ambient-inlet interaction

        # =================================================================
        # Output Variables (3 predictions)
        # =================================================================

        self.UCAOT = 0.0  # Unit Cooler Air Outlet Temperature (°C)
        self.UCWOT = 0.0  # Unit Cooler Water Outlet Temperature (°C)
        self.UCAF = 0.0  # Unit Cooler Air Flow

        # =================================================================
        # Register Variables with FMI
        # =================================================================

        # Register all 52 inputs
        self.register_variable("AMBT", causality=Fmi3Causality.input, variability=Fmi3Variability.continuous)
        self.register_variable("UCTSP", causality=Fmi3Causality.input, variability=Fmi3Variability.continuous)
        self.register_variable("CPSP", causality=Fmi3Causality.input, variability=Fmi3Variability.continuous)
        self.register_variable("UCAIT", causality=Fmi3Causality.input, variability=Fmi3Variability.continuous)
        self.register_variable("CPPR", causality=Fmi3Causality.input, variability=Fmi3Variability.continuous)
        self.register_variable("UCWF", causality=Fmi3Causality.input, variability=Fmi3Variability.continuous)
        self.register_variable("CPMC", causality=Fmi3Causality.input, variability=Fmi3Variability.continuous)
        self.register_variable("MVDP", causality=Fmi3Causality.input, variability=Fmi3Variability.continuous)
        self.register_variable("CPCF", causality=Fmi3Causality.input, variability=Fmi3Variability.continuous)
        self.register_variable("UCFS", causality=Fmi3Causality.input, variability=Fmi3Variability.continuous)
        self.register_variable("MVCV", causality=Fmi3Causality.input, variability=Fmi3Variability.continuous)
        self.register_variable("UCHV", causality=Fmi3Causality.input, variability=Fmi3Variability.continuous)
        self.register_variable("CPMV", causality=Fmi3Causality.input, variability=Fmi3Variability.continuous)
        self.register_variable("UCHC", causality=Fmi3Causality.input, variability=Fmi3Variability.continuous)
        self.register_variable("UCWIT", causality=Fmi3Causality.input, variability=Fmi3Variability.continuous)
        self.register_variable("UCFMS", causality=Fmi3Causality.input, variability=Fmi3Variability.continuous)
        self.register_variable("CPDP", causality=Fmi3Causality.input, variability=Fmi3Variability.continuous)
        self.register_variable("UCWDP", causality=Fmi3Causality.input, variability=Fmi3Variability.continuous)
        self.register_variable("MVWF", causality=Fmi3Causality.input, variability=Fmi3Variability.continuous)
        self.register_variable("UCOM", causality=Fmi3Causality.input, variability=Fmi3Variability.continuous)

        self.register_variable("delta_T_water", causality=Fmi3Causality.input, variability=Fmi3Variability.continuous)
        self.register_variable("delta_T_air", causality=Fmi3Causality.input, variability=Fmi3Variability.continuous)
        self.register_variable("T_approach", causality=Fmi3Causality.input, variability=Fmi3Variability.continuous)
        self.register_variable("T_water_avg", causality=Fmi3Causality.input, variability=Fmi3Variability.continuous)
        self.register_variable("T_air_avg", causality=Fmi3Causality.input, variability=Fmi3Variability.continuous)

        self.register_variable("mdot_water", causality=Fmi3Causality.input, variability=Fmi3Variability.continuous)
        self.register_variable("mdot_air", causality=Fmi3Causality.input, variability=Fmi3Variability.continuous)
        self.register_variable("Q_water", causality=Fmi3Causality.input, variability=Fmi3Variability.continuous)
        self.register_variable("Q_air", causality=Fmi3Causality.input, variability=Fmi3Variability.continuous)
        self.register_variable("Q_avg", causality=Fmi3Causality.input, variability=Fmi3Variability.continuous)
        self.register_variable("Q_imbalance", causality=Fmi3Causality.input, variability=Fmi3Variability.continuous)
        self.register_variable("Q_imbalance_pct", causality=Fmi3Causality.input, variability=Fmi3Variability.continuous)

        self.register_variable("efficiency_HX", causality=Fmi3Causality.input, variability=Fmi3Variability.continuous)
        self.register_variable("effectiveness", causality=Fmi3Causality.input, variability=Fmi3Variability.continuous)
        self.register_variable("NTU", causality=Fmi3Causality.input, variability=Fmi3Variability.continuous)
        self.register_variable("C_ratio", causality=Fmi3Causality.input, variability=Fmi3Variability.continuous)

        self.register_variable("Re_air_estimate", causality=Fmi3Causality.input, variability=Fmi3Variability.continuous)
        self.register_variable("flow_ratio", causality=Fmi3Causality.input, variability=Fmi3Variability.continuous)

        self.register_variable("delta_T_ratio", causality=Fmi3Causality.input, variability=Fmi3Variability.continuous)
        self.register_variable("setpoint_error", causality=Fmi3Causality.input, variability=Fmi3Variability.continuous)
        self.register_variable("setpoint_error_abs", causality=Fmi3Causality.input, variability=Fmi3Variability.continuous)

        self.register_variable("P_fan_estimate", causality=Fmi3Causality.input, variability=Fmi3Variability.continuous)
        self.register_variable("P_pump_estimate", causality=Fmi3Causality.input, variability=Fmi3Variability.continuous)
        self.register_variable("P_total_estimate", causality=Fmi3Causality.input, variability=Fmi3Variability.continuous)
        self.register_variable("COP_estimate", causality=Fmi3Causality.input, variability=Fmi3Variability.continuous)

        self.register_variable("time_index", causality=Fmi3Causality.input, variability=Fmi3Variability.continuous)
        self.register_variable("cycle_hour", causality=Fmi3Causality.input, variability=Fmi3Variability.continuous)
        self.register_variable("hour_sin", causality=Fmi3Causality.input, variability=Fmi3Variability.continuous)
        self.register_variable("hour_cos", causality=Fmi3Causality.input, variability=Fmi3Variability.continuous)

        self.register_variable("T_water_x_flow", causality=Fmi3Causality.input, variability=Fmi3Variability.continuous)
        self.register_variable("T_air_x_flow", causality=Fmi3Causality.input, variability=Fmi3Variability.continuous)
        self.register_variable("ambient_x_inlet", causality=Fmi3Causality.input, variability=Fmi3Variability.continuous)

        # Register 3 outputs
        self.register_variable("UCAOT", causality=Fmi3Causality.output, variability=Fmi3Variability.continuous)
        self.register_variable("UCWOT", causality=Fmi3Causality.output, variability=Fmi3Variability.continuous)
        self.register_variable("UCAF", causality=Fmi3Causality.output, variability=Fmi3Variability.continuous)

        # =================================================================
        # Load ML Models
        # =================================================================

        # Models will be loaded from resources directory
        self.models = None
        self.scaler = None
        self.feature_names = [
            "AMBT", "UCTSP", "CPSP", "UCAIT", "CPPR", "UCWF", "CPMC", "MVDP",
            "CPCF", "UCFS", "MVCV", "UCHV", "CPMV", "UCHC", "UCWIT", "UCFMS",
            "CPDP", "UCWDP", "MVWF", "UCOM", "delta_T_water", "delta_T_air",
            "T_approach", "T_water_avg", "T_air_avg", "mdot_water", "mdot_air",
            "Q_water", "Q_air", "Q_avg", "Q_imbalance", "Q_imbalance_pct",
            "efficiency_HX", "effectiveness", "NTU", "C_ratio", "Re_air_estimate",
            "flow_ratio", "delta_T_ratio", "setpoint_error", "setpoint_error_abs",
            "P_fan_estimate", "P_pump_estimate", "P_total_estimate", "COP_estimate",
            "time_index", "cycle_hour", "hour_sin", "hour_cos", "T_water_x_flow",
            "T_air_x_flow", "ambient_x_inlet"
        ]

    def setup_experiment(self, start_time):
        """Initialize experiment - load models from resources"""
        try:
            # Get resources directory
            resources_dir = Path(self.resources).resolve()

            # Load models
            model_path = resources_dir / "lightgbm_model.pkl"
            scaler_path = resources_dir / "scaler.pkl"

            models_data = joblib.load(str(model_path))
            self.models = models_data['models']
            self.scaler = joblib.load(str(scaler_path))

            self.log(f"Models loaded successfully from {resources_dir}")
            return True
        except Exception as e:
            self.log(f"Error loading models: {e}")
            return False

    def do_step(self, current_time, step_size):
        """
        Perform one simulation step.

        Collects all 52 input features, scales them, and runs predictions
        through the LightGBM models.
        """

        # Skip if models not loaded
        if self.models is None or self.scaler is None:
            return False

        try:
            # Collect all inputs into array (in correct order)
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
            self.log(f"Error in do_step: {e}")
            return False
