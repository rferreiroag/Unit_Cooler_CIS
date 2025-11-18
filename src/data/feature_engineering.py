"""
Physics-based feature engineering for Unit Cooler Digital Twin
Calculates thermodynamic and derived features from raw sensor data
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


class PhysicsFeatureEngineer:
    """Generate physics-informed features for HVAC modeling"""

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

            # Efficiency bounds
            'eta_min': 0.0,
            'eta_max': 1.0,

            # NTU calculation
            'ntu_epsilon': 1e-6,

            # Small value for division safety
            'epsilon': 1e-9
        }

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate all physics-based features

        Args:
            df: Input DataFrame with raw sensor data

        Returns:
            DataFrame with original + engineered features
        """
        print("\n" + "="*80)
        print(" PHYSICS-BASED FEATURE ENGINEERING")
        print("="*80)
        print(f"\nInput features: {len(df.columns)}")

        df_features = df.copy()
        self.feature_names = []

        # Temperature deltas
        df_features = self._add_temperature_deltas(df_features)

        # Mass flow rates
        df_features = self._add_mass_flow_rates(df_features)

        # Thermal power calculations
        df_features = self._add_thermal_power(df_features)

        # Efficiencies and effectiveness
        df_features = self._add_efficiency_metrics(df_features)

        # NTU (Number of Transfer Units)
        df_features = self._add_ntu(df_features)

        # Dimensionless numbers (simplified)
        df_features = self._add_dimensionless_numbers(df_features)

        # Ratios and derived features
        df_features = self._add_ratios(df_features)

        # Power estimates
        df_features = self._add_power_estimates(df_features)

        # Temporal features (if applicable)
        df_features = self._add_temporal_features(df_features)

        # Interaction features
        df_features = self._add_interaction_features(df_features)

        print(f"\nOutput features: {len(df_features.columns)}")
        print(f"Engineered features: {len(self.feature_names)}")
        print(f"\nNew features created:")
        for i, feat in enumerate(self.feature_names, 1):
            print(f"  {i:2d}. {feat}")

        print("="*80)

        return df_features

    def _add_temperature_deltas(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate temperature differences"""
        df_feat = df.copy()

        # Water side temperature delta
        if 'UCWIT' in df.columns and 'UCWOT' in df.columns:
            df_feat['delta_T_water'] = df['UCWIT'] - df['UCWOT']
            self.feature_names.append('delta_T_water')

        # Air side temperature delta
        if 'UCAIT' in df.columns and 'UCAOT' in df.columns:
            df_feat['delta_T_air'] = df['UCAOT'] - df['UCAIT']
            self.feature_names.append('delta_T_air')

        # Temperature approach (water inlet - air inlet)
        if 'UCWIT' in df.columns and 'UCAIT' in df.columns:
            df_feat['T_approach'] = df['UCWIT'] - df['UCAIT']
            self.feature_names.append('T_approach')

        # Average water temperature
        if 'UCWIT' in df.columns and 'UCWOT' in df.columns:
            df_feat['T_water_avg'] = (df['UCWIT'] + df['UCWOT']) / 2.0
            self.feature_names.append('T_water_avg')

        # Average air temperature
        if 'UCAIT' in df.columns and 'UCAOT' in df.columns:
            df_feat['T_air_avg'] = (df['UCAIT'] + df['UCAOT']) / 2.0
            self.feature_names.append('T_air_avg')

        return df_feat

    def _add_mass_flow_rates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert volumetric flows to mass flows"""
        df_feat = df.copy()

        # Water mass flow rate (kg/s)
        # UCWF is in L/min, convert to kg/s
        if 'UCWF' in df.columns:
            # L/min * (1 m³/1000 L) * (1000 kg/m³) * (1 min/60 s) = kg/s
            df_feat['mdot_water'] = df['UCWF'] * self.config['rho_water'] / (self.config['L_per_m3'] * self.config['min_to_s'])
            self.feature_names.append('mdot_water')

        # Air mass flow rate (kg/s)
        # UCAF is in m³/h, convert to kg/s
        if 'UCAF' in df.columns:
            # m³/h * (1.2 kg/m³) * (1 h/3600 s) = kg/s
            df_feat['mdot_air'] = df['UCAF'] * self.config['rho_air'] / 3600.0
            self.feature_names.append('mdot_air')

        return df_feat

    def _add_thermal_power(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate thermal power transfers"""
        df_feat = df.copy()

        # Water side thermal power (kW)
        # Q = m_dot * Cp * delta_T
        if 'mdot_water' in df_feat.columns and 'delta_T_water' in df_feat.columns:
            df_feat['Q_water'] = (df_feat['mdot_water'] *
                                 self.config['Cp_water'] *
                                 df_feat['delta_T_water'] *
                                 self.config['W_to_kW'])
            self.feature_names.append('Q_water')

        # Air side thermal power (kW)
        if 'mdot_air' in df_feat.columns and 'delta_T_air' in df_feat.columns:
            df_feat['Q_air'] = (df_feat['mdot_air'] *
                               self.config['Cp_air'] *
                               df_feat['delta_T_air'] *
                               self.config['W_to_kW'])
            self.feature_names.append('Q_air')

        # Average thermal power (should be similar for energy balance)
        if 'Q_water' in df_feat.columns and 'Q_air' in df_feat.columns:
            df_feat['Q_avg'] = (df_feat['Q_water'].abs() + df_feat['Q_air'].abs()) / 2.0
            self.feature_names.append('Q_avg')

            # Energy imbalance (physics loss indicator)
            df_feat['Q_imbalance'] = df_feat['Q_water'] - df_feat['Q_air']
            self.feature_names.append('Q_imbalance')

            # Relative energy imbalance (%)
            df_feat['Q_imbalance_pct'] = (df_feat['Q_imbalance'] /
                                          (df_feat['Q_avg'].abs() + self.config['epsilon'])) * 100.0
            self.feature_names.append('Q_imbalance_pct')

        return df_feat

    def _add_efficiency_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate heat exchanger efficiency metrics"""
        df_feat = df.copy()

        # Heat exchanger efficiency (Q_air / Q_water)
        if 'Q_air' in df_feat.columns and 'Q_water' in df_feat.columns:
            df_feat['efficiency_HX'] = (df_feat['Q_air'].abs() /
                                        (df_feat['Q_water'].abs() + self.config['epsilon']))

            # Clip to physical bounds [0, 1]
            df_feat['efficiency_HX'] = df_feat['efficiency_HX'].clip(
                self.config['eta_min'], self.config['eta_max']
            )
            self.feature_names.append('efficiency_HX')

        # Effectiveness (actual / maximum possible heat transfer)
        # epsilon = (T_out - T_in) / (T_hot_in - T_cold_in)
        if all(col in df_feat.columns for col in ['delta_T_air', 'UCWIT', 'UCAIT']):
            T_diff_max = df['UCWIT'] - df['UCAIT']
            df_feat['effectiveness'] = (df_feat['delta_T_air'].abs() /
                                       (T_diff_max.abs() + self.config['epsilon']))

            # Clip to [0, 1]
            df_feat['effectiveness'] = df_feat['effectiveness'].clip(0.0, 1.0)
            self.feature_names.append('effectiveness')

        return df_feat

    def _add_ntu(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Number of Transfer Units (NTU)"""
        df_feat = df.copy()

        # NTU from effectiveness: NTU = -ln(1 - epsilon)
        # This is simplified; full NTU depends on flow arrangement
        if 'effectiveness' in df_feat.columns:
            # Avoid log(0) by clipping effectiveness slightly below 1
            eff_clipped = df_feat['effectiveness'].clip(0.0, 0.999)
            df_feat['NTU'] = -np.log(1.0 - eff_clipped + self.config['ntu_epsilon'])

            # Clip to reasonable range
            df_feat['NTU'] = df_feat['NTU'].clip(0.0, 10.0)
            self.feature_names.append('NTU')

        return df_feat

    def _add_dimensionless_numbers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add simplified dimensionless numbers"""
        df_feat = df.copy()

        # Heat capacity rate ratio (C_min / C_max)
        if 'mdot_water' in df_feat.columns and 'mdot_air' in df_feat.columns:
            C_water = df_feat['mdot_water'] * self.config['Cp_water']
            C_air = df_feat['mdot_air'] * self.config['Cp_air']

            C_min = np.minimum(C_water, C_air)
            C_max = np.maximum(C_water, C_air)

            df_feat['C_ratio'] = C_min / (C_max + self.config['epsilon'])
            df_feat['C_ratio'] = df_feat['C_ratio'].clip(0.0, 1.0)
            self.feature_names.append('C_ratio')

        # Reynolds number estimate (simplified)
        # Re ≈ (velocity * characteristic_length) / kinematic_viscosity
        # For air: Re ≈ air_flow^0.8 (empirical correlation)
        if 'UCAF' in df.columns:
            df_feat['Re_air_estimate'] = (df['UCAF'] + 1.0) ** 0.8
            self.feature_names.append('Re_air_estimate')

        return df_feat

    def _add_ratios(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add useful ratio features"""
        df_feat = df.copy()

        # Flow ratio (air / water)
        if 'UCAF' in df.columns and 'UCWF' in df.columns:
            df_feat['flow_ratio'] = (df['UCAF'] + 1.0) / (df['UCWF'] + 1.0)
            self.feature_names.append('flow_ratio')

        # Temperature span ratio
        if 'delta_T_water' in df_feat.columns and 'delta_T_air' in df_feat.columns:
            df_feat['delta_T_ratio'] = ((df_feat['delta_T_air'].abs() + 0.1) /
                                        (df_feat['delta_T_water'].abs() + 0.1))
            self.feature_names.append('delta_T_ratio')

        # Setpoint deviation (if setpoint available)
        if 'UCTSP' in df.columns and 'UCAOT' in df.columns:
            df_feat['setpoint_error'] = df['UCTSP'] - df['UCAOT']
            self.feature_names.append('setpoint_error')

            df_feat['setpoint_error_abs'] = df_feat['setpoint_error'].abs()
            self.feature_names.append('setpoint_error_abs')

        return df_feat

    def _add_power_estimates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Estimate power consumption"""
        df_feat = df.copy()

        # Fan power estimate: P ≈ flow^3 * density / efficiency
        # Simplified: P_fan ≈ (UCAF)^1.5
        if 'UCAF' in df.columns:
            df_feat['P_fan_estimate'] = ((df['UCAF'] + 1.0) ** 1.5) * 0.001  # kW
            self.feature_names.append('P_fan_estimate')

        # Pump power estimate: P = flow * pressure_diff
        # Simplified using water flow
        if 'UCWF' in df.columns:
            df_feat['P_pump_estimate'] = (df['UCWF'] + 1.0) * 0.01  # kW (simplified)
            self.feature_names.append('P_pump_estimate')

        # Total estimated power
        if 'P_fan_estimate' in df_feat.columns and 'P_pump_estimate' in df_feat.columns:
            df_feat['P_total_estimate'] = df_feat['P_fan_estimate'] + df_feat['P_pump_estimate']
            self.feature_names.append('P_total_estimate')

        # COP estimate (Coefficient of Performance)
        if 'Q_avg' in df_feat.columns and 'P_total_estimate' in df_feat.columns:
            df_feat['COP_estimate'] = (df_feat['Q_avg'].abs() /
                                       (df_feat['P_total_estimate'] + self.config['epsilon']))
            df_feat['COP_estimate'] = df_feat['COP_estimate'].clip(0.0, 20.0)
            self.feature_names.append('COP_estimate')

        return df_feat

    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add temporal features based on row index"""
        df_feat = df.copy()

        # Time index (assuming sequential measurements)
        df_feat['time_index'] = np.arange(len(df_feat))
        self.feature_names.append('time_index')

        # Cyclic time features (if sampling rate known, assume 1 sample/min)
        # Hour of cycle (0-23), day of cycle (0-6) - simplified
        minutes_per_day = 1440
        df_feat['cycle_hour'] = (df_feat['time_index'] % minutes_per_day) / 60.0
        self.feature_names.append('cycle_hour')

        # Sine/cosine encoding of hour for cyclical nature
        df_feat['hour_sin'] = np.sin(2 * np.pi * df_feat['cycle_hour'] / 24.0)
        df_feat['hour_cos'] = np.cos(2 * np.pi * df_feat['cycle_hour'] / 24.0)
        self.feature_names.extend(['hour_sin', 'hour_cos'])

        return df_feat

    def _add_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add interaction features between key variables"""
        df_feat = df.copy()

        # Temperature × Flow interactions
        if 'UCWIT' in df.columns and 'UCWF' in df.columns:
            df_feat['T_water_x_flow'] = df['UCWIT'] * df['UCWF']
            self.feature_names.append('T_water_x_flow')

        if 'UCAIT' in df.columns and 'UCAF' in df.columns:
            df_feat['T_air_x_flow'] = df['UCAIT'] * df['UCAF']
            self.feature_names.append('T_air_x_flow')

        # Humidity × Temperature interaction
        if 'UCAIH' in df.columns and 'UCAIT' in df.columns:
            df_feat['humidity_x_temp'] = df['UCAIH'] * df['UCAIT']
            self.feature_names.append('humidity_x_temp')

        # Ambient × Inlet interaction
        if 'AMBT' in df.columns and 'UCAIT' in df.columns:
            df_feat['ambient_x_inlet'] = df['AMBT'] * df['UCAIT']
            self.feature_names.append('ambient_x_inlet')

        return df_feat

    def get_feature_importance_groups(self) -> Dict[str, List[str]]:
        """Group features by category for analysis"""
        groups = {
            'temperature_deltas': [f for f in self.feature_names if 'delta_T' in f or 'T_approach' in f],
            'thermal_power': [f for f in self.feature_names if 'Q_' in f],
            'efficiency': [f for f in self.feature_names if 'efficiency' in f or 'effectiveness' in f or 'COP' in f],
            'flow_derived': [f for f in self.feature_names if 'mdot' in f or 'flow_ratio' in f],
            'dimensionless': [f for f in self.feature_names if any(x in f for x in ['NTU', 'C_ratio', 'Re_'])],
            'power': [f for f in self.feature_names if 'P_' in f],
            'temporal': [f for f in self.feature_names if any(x in f for x in ['time', 'cycle', 'hour'])],
            'interactions': [f for f in self.feature_names if '_x_' in f]
        }

        return groups


def engineer_features(df: pd.DataFrame, config: Optional[Dict] = None) -> Tuple[pd.DataFrame, PhysicsFeatureEngineer]:
    """
    Convenience function to engineer physics-based features

    Args:
        df: Input DataFrame (preprocessed)
        config: Optional configuration

    Returns:
        Tuple of (DataFrame with features, feature engineer instance)
    """
    engineer = PhysicsFeatureEngineer(config=config)
    df_features = engineer.fit_transform(df)

    return df_features, engineer


if __name__ == "__main__":
    # Test feature engineering
    import sys
    sys.path.append('../..')
    from src.data.data_loader import load_and_preprocess
    from src.data.preprocessing import preprocess_unit_cooler_data

    # Load and preprocess data
    df, metadata = load_and_preprocess('../../data/raw/datos_combinados_entrenamiento_20251118_105234.csv')
    df_clean, preprocessor = preprocess_unit_cooler_data(df)

    # Engineer features
    df_features, engineer = engineer_features(df_clean)

    print("\n" + "="*80)
    print(" FEATURE ENGINEERING COMPLETE")
    print("="*80)
    print(f"Original features: {len(df_clean.columns)}")
    print(f"Final features: {len(df_features.columns)}")
    print(f"New features: {len(engineer.feature_names)}")

    # Feature groups
    groups = engineer.get_feature_importance_groups()
    print("\nFeature groups:")
    for group_name, features in groups.items():
        print(f"  {group_name}: {len(features)} features")
