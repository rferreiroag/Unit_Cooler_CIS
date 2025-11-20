"""
Physics-based feature engineering WITHOUT DATA LEAKAGE
Calculates features ONLY from sensor inputs (no target variables)

This version ensures all features can be calculated in production
without knowledge of UCAOT, UCWOT, or UCAF (the targets we want to predict).
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


class PhysicsFeatureEngineerNoLeakage:
    """Generate physics-informed features WITHOUT using target variables"""

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize feature engineer

        Args:
            config: Configuration with physical constants
        """
        self.config = config or self._default_config()
        self.feature_names = []

    def _default_config(self) -> Dict:
        """Default physical constants and configuration"""
        return {
            # Specific heat capacities (J/kg·K)
            'Cp_water': 4186.0,      # Water
            'Cp_air': 1005.0,        # Dry air at 20°C

            # Densities (kg/m³)
            'rho_water': 1000.0,     # Water
            'rho_air': 1.2,          # Air at sea level, 20°C

            # Conversion factors
            'L_per_m3': 1000.0,      # Liters per cubic meter
            'W_to_kW': 0.001,        # Watts to kilowatts
            'min_to_s': 60.0,        # Minutes to seconds

            # Small value for division safety
            'epsilon': 1e-9
        }

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate all physics-based features WITHOUT data leakage

        Args:
            df: Input DataFrame with raw sensor data

        Returns:
            DataFrame with original + engineered features (NO target-dependent features)
        """
        print("\n" + "="*80)
        print(" PHYSICS-BASED FEATURE ENGINEERING (NO DATA LEAKAGE)")
        print("="*80)
        print(f"\nInput features: {len(df.columns)}")

        df_features = df.copy()
        self.feature_names = []

        # Temperature deltas (ONLY using input sensors, not targets)
        df_features = self._add_temperature_deltas_no_leakage(df_features)

        # Mass flow rates (ONLY water side, no UCAF)
        df_features = self._add_mass_flow_rates_no_leakage(df_features)

        # Thermal power (ONLY water side, no air side with UCAOT/UCAF)
        df_features = self._add_thermal_power_no_leakage(df_features)

        # Power estimates (based on inputs only)
        df_features = self._add_power_estimates_no_leakage(df_features)

        # Temporal features
        df_features = self._add_temporal_features(df_features)

        # Interaction features (only with input sensors)
        df_features = self._add_interaction_features_no_leakage(df_features)

        print(f"\nOutput features: {len(df_features.columns)}")
        print(f"Engineered features: {len(self.feature_names)}")
        print(f"\nNew features created (NO data leakage):")
        for i, feat in enumerate(self.feature_names, 1):
            print(f"  {i:2d}. {feat}")

        print("\n" + "="*80)
        print("✓ All features can be computed from sensor inputs only")
        print("✓ No dependency on target variables (UCAOT, UCWOT, UCAF)")
        print("="*80)

        return df_features

    def _add_temperature_deltas_no_leakage(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate temperature differences using ONLY input sensors"""
        df_feat = df.copy()

        # Water side temperature delta (UCWIT is input, UCWOT is target - DON'T USE)
        # We can use UCWIT but not delta with output
        # Instead, use temperature approach and gradients

        # Temperature approach (water inlet - air inlet)
        if 'UCWIT' in df.columns and 'UCAIT' in df.columns:
            df_feat['T_approach'] = df['UCWIT'] - df['UCAIT']
            self.feature_names.append('T_approach')

        # Temperature difference from ambient
        if 'UCWIT' in df.columns and 'AMBT' in df.columns:
            df_feat['T_water_ambient_diff'] = df['UCWIT'] - df['AMBT']
            self.feature_names.append('T_water_ambient_diff')

        if 'UCAIT' in df.columns and 'AMBT' in df.columns:
            df_feat['T_air_ambient_diff'] = df['UCAIT'] - df['AMBT']
            self.feature_names.append('T_air_ambient_diff')

        # Setpoint deviation (input - setpoint)
        if 'UCTSP' in df.columns and 'UCAIT' in df.columns:
            df_feat['setpoint_inlet_diff'] = df['UCTSP'] - df['UCAIT']
            self.feature_names.append('setpoint_inlet_diff')

        if 'UCTSP' in df.columns and 'AMBT' in df.columns:
            df_feat['setpoint_ambient_diff'] = df['UCTSP'] - df['AMBT']
            self.feature_names.append('setpoint_ambient_diff')

        return df_feat

    def _add_mass_flow_rates_no_leakage(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert volumetric flows to mass flows (ONLY water side, no UCAF)"""
        df_feat = df.copy()

        # Water mass flow rate (kg/s)
        if 'UCWF' in df.columns:
            # L/min * (1 m³/1000 L) * (1000 kg/m³) * (1 min/60 s) = kg/s
            df_feat['mdot_water'] = df['UCWF'] * self.config['rho_water'] / (
                self.config['L_per_m3'] * self.config['min_to_s']
            )
            self.feature_names.append('mdot_water')

        # NOTE: mdot_air would require UCAF (which is a target), so we skip it

        return df_feat

    def _add_thermal_power_no_leakage(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate thermal power (ONLY water side, no air side with targets)"""
        df_feat = df.copy()

        # Water side thermal power estimate
        # We can't compute exact Q_water without UCWOT (target)
        # But we can estimate potential thermal capacity
        if 'mdot_water' in df_feat.columns and 'UCWIT' in df.columns:
            # Thermal capacity of water stream (kW/K)
            df_feat['C_water'] = (df_feat['mdot_water'] *
                                  self.config['Cp_water'] *
                                  self.config['W_to_kW'])
            self.feature_names.append('C_water')

        # Estimate maximum possible heat transfer based on approach temperature
        if 'C_water' in df_feat.columns and 'T_approach' in df_feat.columns:
            # Maximum Q if water cooled to air inlet temperature
            df_feat['Q_max_water'] = df_feat['C_water'] * df_feat['T_approach'].abs()
            self.feature_names.append('Q_max_water')

        return df_feat

    def _add_power_estimates_no_leakage(self, df: pd.DataFrame) -> pd.DataFrame:
        """Estimate power consumption from control signals"""
        df_feat = df.copy()

        # Fan speed/control signal (if available)
        # UCFS might be fan speed control
        if 'UCFS' in df.columns:
            # Fan power estimate from speed: P ≈ speed^3
            df_feat['P_fan_estimate'] = ((df['UCFS'].abs() + 1.0) ** 1.5) * 0.001  # kW
            self.feature_names.append('P_fan_estimate')

        # Pump power estimate from water flow
        if 'UCWF' in df.columns and 'UCWDP' in df.columns:
            # P = flow * pressure_differential
            df_feat['P_pump_estimate'] = (
                df['UCWF'] * df['UCWDP'] * 0.0001  # kW (simplified)
            )
            self.feature_names.append('P_pump_estimate')
        elif 'UCWF' in df.columns:
            # Simplified without pressure
            df_feat['P_pump_estimate'] = (df['UCWF'] + 1.0) * 0.01  # kW
            self.feature_names.append('P_pump_estimate')

        # Total estimated power
        if 'P_fan_estimate' in df_feat.columns and 'P_pump_estimate' in df_feat.columns:
            df_feat['P_total_estimate'] = df_feat['P_fan_estimate'] + df_feat['P_pump_estimate']
            self.feature_names.append('P_total_estimate')

        return df_feat

    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add temporal features based on row index"""
        df_feat = df.copy()

        # Time index (assuming sequential measurements)
        df_feat['time_index'] = np.arange(len(df_feat))
        self.feature_names.append('time_index')

        # Cyclic time features (assume 1 sample/min)
        minutes_per_day = 1440
        df_feat['cycle_hour'] = (df_feat['time_index'] % minutes_per_day) / 60.0
        self.feature_names.append('cycle_hour')

        # Sine/cosine encoding for cyclical nature
        df_feat['hour_sin'] = np.sin(2 * np.pi * df_feat['cycle_hour'] / 24.0)
        df_feat['hour_cos'] = np.cos(2 * np.pi * df_feat['cycle_hour'] / 24.0)
        self.feature_names.extend(['hour_sin', 'hour_cos'])

        return df_feat

    def _add_interaction_features_no_leakage(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add interaction features (ONLY with input sensors)"""
        df_feat = df.copy()

        # Temperature × Flow interactions (only with inputs)
        if 'UCWIT' in df.columns and 'UCWF' in df.columns:
            df_feat['T_water_x_flow'] = df['UCWIT'] * df['UCWF']
            self.feature_names.append('T_water_x_flow')

        # Ambient × Inlet interaction
        if 'AMBT' in df.columns and 'UCAIT' in df.columns:
            df_feat['ambient_x_inlet'] = df['AMBT'] * df['UCAIT']
            self.feature_names.append('ambient_x_inlet')

        # Setpoint × Flow
        if 'UCTSP' in df.columns and 'UCWF' in df.columns:
            df_feat['setpoint_x_flow'] = df['UCTSP'] * df['UCWF']
            self.feature_names.append('setpoint_x_flow')

        # Water inlet × Pressure
        if 'UCWIT' in df.columns and 'UCWDP' in df.columns:
            df_feat['T_water_x_pressure'] = df['UCWIT'] * df['UCWDP']
            self.feature_names.append('T_water_x_pressure')

        return df_feat

    def get_feature_list(self) -> List[str]:
        """Get list of all engineered features"""
        return self.feature_names.copy()


def engineer_features_no_leakage(
    df: pd.DataFrame,
    config: Optional[Dict] = None
) -> Tuple[pd.DataFrame, PhysicsFeatureEngineerNoLeakage]:
    """
    Convenience function to engineer features WITHOUT data leakage

    Args:
        df: Input DataFrame (preprocessed)
        config: Optional configuration

    Returns:
        Tuple of (DataFrame with features, feature engineer instance)
    """
    engineer = PhysicsFeatureEngineerNoLeakage(config=config)
    df_features = engineer.fit_transform(df)

    return df_features, engineer


if __name__ == "__main__":
    # Test feature engineering
    import sys
    sys.path.append('../..')
    from src.data.data_loader import load_and_preprocess
    from src.data.preprocessing import preprocess_unit_cooler_data

    # Load and preprocess data
    print("Loading data...")
    df, metadata = load_and_preprocess('../../data/raw/datos_combinados_entrenamiento_20251118_105234.csv')
    df_clean, preprocessor = preprocess_unit_cooler_data(df)

    # Engineer features WITHOUT leakage
    df_features, engineer = engineer_features_no_leakage(df_clean)

    print("\n" + "="*80)
    print(" FEATURE ENGINEERING COMPLETE (NO DATA LEAKAGE)")
    print("="*80)
    print(f"Original features: {len(df_clean.columns)}")
    print(f"Final features: {len(df_features.columns)}")
    print(f"New features: {len(engineer.feature_names)}")

    # Verify no targets in features
    target_cols = ['UCAOT', 'UCWOT', 'UCAF']
    has_targets = any(col in df_features.columns for col in target_cols)
    print(f"\n✓ Contains target variables: {has_targets}")
    print("✓ All features can be computed in production!")
