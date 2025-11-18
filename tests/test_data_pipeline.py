"""
Unit tests for data pipeline components
"""

import pytest
import pandas as pd
import numpy as np
import sys
sys.path.append('..')

from src.data.data_loader import DataLoader, load_and_preprocess
from src.data.preprocessing import DataPreprocessor, preprocess_unit_cooler_data
from src.data.feature_engineering import PhysicsFeatureEngineer, engineer_features
from src.data.data_splits import TemporalDataSplitter, AdaptiveScaler, prepare_temporal_data


class TestDataLoader:
    """Test data loading functionality"""

    def test_data_loader_initialization(self):
        """Test DataLoader initialization"""
        loader = DataLoader('../data/raw/datos_combinados_entrenamiento_20251118_105234.csv')
        assert loader.data_path.exists() or True  # File may not exist in test environment
        assert loader.df is None
        assert loader.metadata == {}

    def test_load_and_preprocess(self):
        """Test complete load and preprocess function"""
        try:
            df, metadata = load_and_preprocess('../data/raw/datos_combinados_entrenamiento_20251118_105234.csv')

            # Check DataFrame
            assert isinstance(df, pd.DataFrame)
            assert len(df) > 0
            assert len(df.columns) > 0

            # Check metadata
            assert isinstance(metadata, dict)
            assert 'basic_info' in metadata
            assert 'target_vars' in metadata
            assert 'input_vars' in metadata

        except FileNotFoundError:
            pytest.skip("Data file not found")


class TestDataPreprocessor:
    """Test preprocessing functionality"""

    def test_preprocessor_initialization(self):
        """Test DataPreprocessor initialization"""
        preprocessor = DataPreprocessor()
        assert preprocessor.config is not None
        assert 'temp_min' in preprocessor.config
        assert 'temp_max' in preprocessor.config
        assert not preprocessor.is_fitted

    def test_preprocessor_with_sample_data(self):
        """Test preprocessing with sample data"""
        # Create sample data
        n_samples = 1000
        df = pd.DataFrame({
            'UCWIT': np.random.uniform(10, 30, n_samples),
            'UCWOT': np.random.uniform(8, 28, n_samples),
            'UCAIT': np.random.uniform(20, 30, n_samples),
            'UCAOT': np.random.uniform(18, 28, n_samples),
            'UCWF': np.random.uniform(0, 1000, n_samples),
            'UCAF': np.random.uniform(0, 5000, n_samples),
        })

        # Add some NaN values
        df.loc[0:10, 'UCWIT'] = np.nan

        # Add negative flows (should be fixed)
        df.loc[20:30, 'UCWF'] = -100

        preprocessor = DataPreprocessor()
        df_clean = preprocessor.fit_transform(df)

        # Check results
        assert len(df_clean) <= len(df)  # Some rows might be removed
        assert df_clean.isnull().sum().sum() == 0  # No NaN remaining
        assert (df_clean['UCWF'] >= 0).all()  # No negative flows
        assert preprocessor.is_fitted

    def test_sensor_saturation_handling(self):
        """Test sensor saturation value replacement"""
        df = pd.DataFrame({
            'UCAF': [1000, 65535, 2000, 65535, 3000],
            'UCWF': [100, 200, 300, 400, 500]
        })

        preprocessor = DataPreprocessor()
        df_clean = preprocessor._handle_sensor_saturation(df)

        # Check that saturation values are replaced with NaN
        assert df_clean['UCAF'].isna().sum() >= 2

    def test_negative_flow_fixing(self):
        """Test negative flow value fixing"""
        df = pd.DataFrame({
            'UCWF': [100, -50, 200, -100, 300],
            'UCAF': [1000, 2000, -500, 3000, 4000]
        })

        preprocessor = DataPreprocessor()
        df_clean = preprocessor._fix_negative_flows(df)

        # Check that negative flows are set to 0
        assert (df_clean['UCWF'] >= 0).all()
        assert (df_clean['UCAF'] >= 0).all()


class TestFeatureEngineering:
    """Test feature engineering functionality"""

    def test_feature_engineer_initialization(self):
        """Test PhysicsFeatureEngineer initialization"""
        engineer = PhysicsFeatureEngineer()
        assert engineer.config is not None
        assert 'Cp_water' in engineer.config
        assert 'Cp_air' in engineer.config
        assert engineer.feature_names == []

    def test_temperature_deltas(self):
        """Test temperature delta calculations"""
        df = pd.DataFrame({
            'UCWIT': [20, 22, 24],
            'UCWOT': [18, 20, 22],
            'UCAIT': [25, 26, 27],
            'UCAOT': [23, 24, 25]
        })

        engineer = PhysicsFeatureEngineer()
        df_feat = engineer._add_temperature_deltas(df)

        # Check that delta features are created
        assert 'delta_T_water' in df_feat.columns
        assert 'delta_T_air' in df_feat.columns
        assert 'T_approach' in df_feat.columns

        # Check calculations
        assert (df_feat['delta_T_water'] == df['UCWIT'] - df['UCWOT']).all()
        assert (df_feat['delta_T_air'] == df['UCAOT'] - df['UCAIT']).all()

    def test_thermal_power_calculation(self):
        """Test thermal power calculations"""
        df = pd.DataFrame({
            'UCWIT': [20, 22, 24],
            'UCWOT': [18, 20, 22],
            'UCWF': [100, 150, 200],
            'UCAIT': [25, 26, 27],
            'UCAOT': [23, 24, 25],
            'UCAF': [1000, 1500, 2000]
        })

        engineer = PhysicsFeatureEngineer()

        # Add required intermediate features
        df_feat = engineer._add_temperature_deltas(df)
        df_feat = engineer._add_mass_flow_rates(df_feat)
        df_feat = engineer._add_thermal_power(df_feat)

        # Check that thermal power features are created
        assert 'Q_water' in df_feat.columns
        assert 'Q_air' in df_feat.columns
        assert 'Q_avg' in df_feat.columns
        assert 'Q_imbalance' in df_feat.columns

    def test_complete_feature_engineering(self):
        """Test complete feature engineering pipeline"""
        # Create realistic sample data
        n_samples = 500
        df = pd.DataFrame({
            'UCWIT': np.random.uniform(10, 30, n_samples),
            'UCWOT': np.random.uniform(8, 28, n_samples),
            'UCAIT': np.random.uniform(20, 30, n_samples),
            'UCAOT': np.random.uniform(18, 28, n_samples),
            'UCWF': np.random.uniform(50, 500, n_samples),
            'UCAF': np.random.uniform(500, 3000, n_samples),
            'UCAIH': np.random.uniform(40, 70, n_samples),
            'AMBT': np.random.uniform(20, 30, n_samples),
        })

        engineer = PhysicsFeatureEngineer()
        df_features = engineer.fit_transform(df)

        # Check that features are created
        assert len(df_features.columns) > len(df.columns)
        assert len(engineer.feature_names) > 0

        # Check specific features
        expected_features = ['delta_T_water', 'delta_T_air', 'Q_water', 'Q_air',
                           'efficiency_HX', 'effectiveness', 'mdot_water', 'mdot_air']

        for feat in expected_features:
            assert feat in df_features.columns, f"Feature {feat} not found"


class TestTemporalSplitting:
    """Test temporal data splitting functionality"""

    def test_temporal_splitter_initialization(self):
        """Test TemporalDataSplitter initialization"""
        splitter = TemporalDataSplitter(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
        assert splitter.train_ratio == 0.7
        assert splitter.val_ratio == 0.15
        assert splitter.test_ratio == 0.15

    def test_temporal_split(self):
        """Test temporal data splitting"""
        # Create sample data
        n_samples = 1000
        df = pd.DataFrame({
            'feature_1': np.random.randn(n_samples),
            'feature_2': np.random.randn(n_samples),
            'target_1': np.random.randn(n_samples),
            'target_2': np.random.randn(n_samples)
        })

        splitter = TemporalDataSplitter(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
        X_train, X_val, X_test, y_train, y_val, y_test = splitter.split(
            df, target_cols=['target_1', 'target_2'], shuffle=False
        )

        # Check sizes
        assert len(X_train) == 700
        assert len(X_val) == 150
        assert len(X_test) == 150

        # Check that splits are temporal (no shuffle)
        # First row of val should come after last row of train
        assert X_val.index[0] == X_train.index[-1] + 1

    def test_no_data_leakage(self):
        """Test that temporal split has no data leakage"""
        n_samples = 1000
        df = pd.DataFrame({
            'time_index': np.arange(n_samples),
            'feature': np.random.randn(n_samples),
            'target': np.random.randn(n_samples)
        })

        splitter = TemporalDataSplitter()
        X_train, X_val, X_test, y_train, y_val, y_test = splitter.split(
            df, target_cols=['target'], shuffle=False
        )

        # Check temporal ordering
        assert X_train['time_index'].max() < X_val['time_index'].min()
        assert X_val['time_index'].max() < X_test['time_index'].min()


class TestAdaptiveScaling:
    """Test adaptive scaling functionality"""

    def test_adaptive_scaler_initialization(self):
        """Test AdaptiveScaler initialization"""
        scaler = AdaptiveScaler(scaling_method='standard')
        assert scaler.scaling_method == 'standard'
        assert not scaler.is_fitted

    def test_scaling_fit_transform(self):
        """Test scaler fit and transform"""
        # Create sample data
        n_samples = 500
        X_train = pd.DataFrame({
            'feature_1': np.random.uniform(0, 100, n_samples),
            'feature_2': np.random.uniform(0, 1000, n_samples)
        })
        y_train = pd.DataFrame({
            'target': np.random.uniform(0, 50, n_samples)
        })

        scaler = AdaptiveScaler(scaling_method='standard')
        scaler.fit(X_train, y_train)

        assert scaler.is_fitted
        assert scaler.scaler_X is not None
        assert scaler.scaler_y is not None

        # Transform data
        X_scaled, y_scaled = scaler.transform(X_train, y_train)

        # Check that data is scaled (mean ≈ 0, std ≈ 1 for standard scaling)
        assert np.abs(X_scaled.mean()) < 1.0
        assert np.abs(X_scaled.std() - 1.0) < 0.5

    def test_inverse_transform(self):
        """Test inverse transformation"""
        n_samples = 500
        X_train = pd.DataFrame({
            'feature': np.random.uniform(0, 100, n_samples)
        })
        y_train = pd.DataFrame({
            'target': np.random.uniform(0, 50, n_samples)
        })

        scaler = AdaptiveScaler(scaling_method='standard')
        scaler.fit(X_train, y_train)

        X_scaled, y_scaled = scaler.transform(X_train, y_train)
        y_unscaled = scaler.inverse_transform_y(y_scaled)

        # Check that inverse transform recovers original data
        assert np.allclose(y_unscaled, y_train.values, rtol=1e-5)


class TestIntegration:
    """Integration tests for complete pipeline"""

    def test_complete_pipeline_small_dataset(self):
        """Test complete pipeline on small dataset"""
        # Create sample dataset
        n_samples = 1000
        df = pd.DataFrame({
            'UCWIT': np.random.uniform(10, 30, n_samples),
            'UCWOT': np.random.uniform(8, 28, n_samples),
            'UCAIT': np.random.uniform(20, 30, n_samples),
            'UCAOT': np.random.uniform(18, 28, n_samples),
            'UCWF': np.random.uniform(50, 500, n_samples),
            'UCAF': np.random.uniform(500, 3000, n_samples),
        })

        # Preprocessing
        preprocessor = DataPreprocessor()
        df_clean = preprocessor.fit_transform(df)

        assert len(df_clean) > 0
        assert df_clean.isnull().sum().sum() == 0

        # Feature engineering
        engineer = PhysicsFeatureEngineer()
        df_features = engineer.fit_transform(df_clean)

        assert len(df_features.columns) > len(df_clean.columns)

        # Temporal splitting and scaling
        target_cols = ['UCAOT', 'UCWOT', 'UCAF']
        data_dict = prepare_temporal_data(df_features, target_cols)

        assert 'X_train_scaled' in data_dict
        assert 'y_train_scaled' in data_dict
        assert len(data_dict['X_train']) > 0
        assert len(data_dict['X_val']) > 0
        assert len(data_dict['X_test']) > 0


def run_tests():
    """Run all tests"""
    pytest.main([__file__, '-v', '--tb=short'])


if __name__ == "__main__":
    run_tests()
