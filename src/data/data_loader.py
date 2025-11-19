"""
Data loading and preprocessing module for Unit Cooler Digital Twin
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Optional
import warnings
warnings.filterwarnings('ignore')


class DataLoader:
    """Load and validate Unit Cooler experimental data"""

    def __init__(self, data_path: str):
        """
        Initialize DataLoader

        Args:
            data_path: Path to CSV data file
        """
        self.data_path = Path(data_path)
        self.df = None
        self.metadata = {}

    def load_data(self) -> pd.DataFrame:
        """
        Load CSV data with validation

        Returns:
            DataFrame with loaded data
        """
        print(f"Loading data from: {self.data_path}")

        try:
            # Try to detect delimiter and load data
            # CSV file uses semicolon as delimiter
            self.df = pd.read_csv(self.data_path, sep=';', skipinitialspace=True)

            # Remove empty rows if any
            self.df = self.df.dropna(how='all')

            # Reset index
            self.df = self.df.reset_index(drop=True)

            print(f"✓ Data loaded successfully")
            print(f"  Shape: {self.df.shape}")
            print(f"  Columns: {len(self.df.columns)}")
            print(f"  Rows: {len(self.df)}")

            self._extract_metadata()
            return self.df

        except FileNotFoundError:
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        except Exception as e:
            raise Exception(f"Error loading data: {str(e)}")

    def _extract_metadata(self):
        """Extract metadata from loaded data"""
        if self.df is not None:
            self.metadata = {
                'n_rows': len(self.df),
                'n_cols': len(self.df.columns),
                'columns': list(self.df.columns),
                'dtypes': self.df.dtypes.to_dict(),
                'memory_usage_mb': self.df.memory_usage(deep=True).sum() / 1024**2
            }

    def get_basic_info(self) -> Dict:
        """
        Get basic dataset information

        Returns:
            Dictionary with dataset statistics
        """
        if self.df is None:
            raise ValueError("No data loaded. Call load_data() first.")

        info = {
            'shape': self.df.shape,
            'columns': list(self.df.columns),
            'dtypes': self.df.dtypes.value_counts().to_dict(),
            'memory_usage_mb': round(self.metadata['memory_usage_mb'], 2),
            'missing_values': self.df.isnull().sum().to_dict(),
            'missing_percentage': (self.df.isnull().sum() / len(self.df) * 100).round(2).to_dict(),
            'duplicated_rows': self.df.duplicated().sum()
        }

        return info

    def get_statistics(self) -> pd.DataFrame:
        """
        Get descriptive statistics for numeric columns

        Returns:
            DataFrame with statistics
        """
        if self.df is None:
            raise ValueError("No data loaded. Call load_data() first.")

        return self.df.describe()

    def check_data_quality(self) -> Dict:
        """
        Comprehensive data quality check

        Returns:
            Dictionary with quality metrics
        """
        if self.df is None:
            raise ValueError("No data loaded. Call load_data() first.")

        quality = {}

        # Missing values
        missing = self.df.isnull().sum()
        quality['columns_with_missing'] = missing[missing > 0].to_dict()
        quality['total_missing_cells'] = self.df.isnull().sum().sum()
        quality['missing_percentage_overall'] = round(
            self.df.isnull().sum().sum() / (self.df.shape[0] * self.df.shape[1]) * 100, 2
        )

        # Duplicates
        quality['duplicated_rows'] = self.df.duplicated().sum()

        # Data types
        quality['dtypes_summary'] = self.df.dtypes.value_counts().to_dict()

        # Numeric columns statistics
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        quality['n_numeric_columns'] = len(numeric_cols)
        quality['n_categorical_columns'] = len(self.df.columns) - len(numeric_cols)

        # Check for constant columns
        constant_cols = [col for col in self.df.columns if self.df[col].nunique() <= 1]
        quality['constant_columns'] = constant_cols

        # Check for infinite values in numeric columns
        inf_counts = {}
        for col in numeric_cols:
            inf_count = np.isinf(self.df[col]).sum()
            if inf_count > 0:
                inf_counts[col] = inf_count
        quality['columns_with_inf'] = inf_counts

        return quality

    def identify_target_variables(self) -> Dict[str, bool]:
        """
        Identify key target variables in dataset

        Returns:
            Dictionary with target variable presence
        """
        target_vars = {
            'UCAOT': 'UCAOT' in self.df.columns,  # Unit Cooler Air Outlet Temperature
            'UCWOT': 'UCWOT' in self.df.columns,  # Unit Cooler Water Outlet Temperature
            'UCAF': 'UCAF' in self.df.columns,    # Unit Cooler Air Flow
            'Q_thermal': 'Q_thermal' in self.df.columns  # Thermal power
        }

        return target_vars

    def identify_input_variables(self) -> Dict[str, bool]:
        """
        Identify key input variables in dataset

        Returns:
            Dictionary with input variable presence
        """
        input_vars = {
            'UCWIT': 'UCWIT' in self.df.columns,  # Water Inlet Temperature
            'UCAIT': 'UCAIT' in self.df.columns,  # Air Inlet Temperature
            'UCWF': 'UCWF' in self.df.columns,    # Water Flow
            'UCAF': 'UCAF' in self.df.columns,    # Air Flow
            'UCAIH': 'UCAIH' in self.df.columns,  # Air Inlet Humidity
            'AMBT': 'AMBT' in self.df.columns,    # Ambient Temperature
            'AMBH': 'AMBH' in self.df.columns     # Ambient Humidity
        }

        return input_vars


def load_and_preprocess(data_path: str) -> Tuple[pd.DataFrame, Dict]:
    """
    Main function to load and perform initial preprocessing

    Args:
        data_path: Path to CSV data file

    Returns:
        Tuple of (DataFrame, metadata dictionary)
    """
    loader = DataLoader(data_path)
    df = loader.load_data()

    metadata = {
        'basic_info': loader.get_basic_info(),
        'quality': loader.check_data_quality(),
        'target_vars': loader.identify_target_variables(),
        'input_vars': loader.identify_input_variables()
    }

    return df, metadata


if __name__ == "__main__":
    # Test the data loader
    data_path = "../../data/raw/datos_combinados_entrenamiento_20251118_105234.csv"
    df, metadata = load_and_preprocess(data_path)

    print("\n=== Dataset Information ===")
    print(f"Shape: {metadata['basic_info']['shape']}")
    print(f"Memory: {metadata['basic_info']['memory_usage_mb']} MB")
    print(f"\nTarget variables present: {metadata['target_vars']}")
    print(f"Input variables present: {metadata['input_vars']}")
