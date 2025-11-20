"""
HVAC Unit Cooler Digital Twin FMU

This FMU implements the LightGBM-based digital twin for HVAC unit cooler prediction.

Inputs: 52 features (sensor measurements and engineered features)
Outputs: 3 predictions (UCAOT, UCWOT, UCAF)

To build this FMU:
    pip install pythonfmu
    pythonfmu build -f deployment/fmu/hvac_unit_cooler_fmu.py
"""

from pythonfmu import Fmi2Causality, Fmi2Slave, Fmi2Variability, Real
import numpy as np
import joblib
from pathlib import Path


class HVACUnitCoolerFMU(Fmi2Slave):

    author = "HVAC Digital Twin Team"
    description = "LightGBM ML model for HVAC unit cooler prediction"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Load models and preprocessing (from resources directory)
        resources_dir = Path(__file__).parent / "resources"

        models_data = joblib.load(resources_dir / "lightgbm_model.pkl")
        self.models = models_data['models']
        self.scaler = joblib.load(resources_dir / "scaler.pkl")

        # Define input variables (52 features)
        self.ambt = 0.0
        self.register_variable(Real("AMBT", causality=Fmi2Causality.input, variability=Fmi2Variability.continuous))
        self.uctsp = 0.0
        self.register_variable(Real("UCTSP", causality=Fmi2Causality.input, variability=Fmi2Variability.continuous))
        self.cpsp = 0.0
        self.register_variable(Real("CPSP", causality=Fmi2Causality.input, variability=Fmi2Variability.continuous))
        self.ucait = 0.0
        self.register_variable(Real("UCAIT", causality=Fmi2Causality.input, variability=Fmi2Variability.continuous))
        self.cppr = 0.0
        self.register_variable(Real("CPPR", causality=Fmi2Causality.input, variability=Fmi2Variability.continuous))
        self.ucwf = 0.0
        self.register_variable(Real("UCWF", causality=Fmi2Causality.input, variability=Fmi2Variability.continuous))
        self.cpmc = 0.0
        self.register_variable(Real("CPMC", causality=Fmi2Causality.input, variability=Fmi2Variability.continuous))
        self.mvdp = 0.0
        self.register_variable(Real("MVDP", causality=Fmi2Causality.input, variability=Fmi2Variability.continuous))
        self.cpcf = 0.0
        self.register_variable(Real("CPCF", causality=Fmi2Causality.input, variability=Fmi2Variability.continuous))
        self.ucfs = 0.0
        self.register_variable(Real("UCFS", causality=Fmi2Causality.input, variability=Fmi2Variability.continuous))
        # ... (register all 52 input features)

        # Define output variables (3 targets)
        self.UCAOT = 0.0
        self.UCWOT = 0.0
        self.UCAF = 0.0

        self.register_variable(Real("UCAOT", causality=Fmi2Causality.output, variability=Fmi2Variability.continuous))
        self.register_variable(Real("UCWOT", causality=Fmi2Causality.output, variability=Fmi2Variability.continuous))
        self.register_variable(Real("UCAF", causality=Fmi2Causality.output, variability=Fmi2Variability.continuous))

    def do_step(self, current_time, step_size):
        # Collect all inputs into array
        inputs = np.array([
            # Collect all 52 feature values
            # self.feature1, self.feature2, ..., self.feature52
        ])

        # Scale inputs
        X_scaled = self.scaler.transform(inputs.reshape(1, -1))

        # Make predictions
        self.UCAOT = float(self.models['UCAOT'].predict(X_scaled)[0])
        self.UCWOT = float(self.models['UCWOT'].predict(X_scaled)[0])
        self.UCAF = float(self.models['UCAF'].predict(X_scaled)[0])

        return True
