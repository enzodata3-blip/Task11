"""
Data Processing Module
Load, clean, and prepare data for analysis and modeling.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Dict, Any
import warnings
warnings.filterwarnings('ignore')


class DataProcessor:
    """
    Handle data loading, cleaning, and preprocessing.

    Emphasis on reproducibility and data quality checks.
    """

    def __init__(self):
        """Initialize data processor."""
        self.data = None
        self.data_info = {}

    def load_data(self, filepath: str, **kwargs) -> pd.DataFrame:
        """
        Load data from various file formats.

        Parameters:
        -----------
        filepath : str
            Path to data file
        **kwargs : additional arguments passed to pandas reader

        Returns:
        --------
        pd.DataFrame
            Loaded data
        """
        print(f"Loading data from {filepath}...")

        if filepath.endswith('.csv'):
            self.data = pd.read_csv(filepath, **kwargs)
        elif filepath.endswith('.xlsx') or filepath.endswith('.xls'):
            self.data = pd.read_excel(filepath, **kwargs)
        elif filepath.endswith('.parquet'):
            self.data = pd.read_parquet(filepath, **kwargs)
        elif filepath.endswith('.json'):
            self.data = pd.read_json(filepath, **kwargs)
        else:
            raise ValueError(f"Unsupported file format: {filepath}")

        print(f"Data loaded: {self.data.shape[0]} rows, {self.data.shape[1]} columns")

        return self.data

    def generate_data_profile(self) -> Dict:
        """
        Generate comprehensive data profile.

        Returns:
        --------
        Dict
            Data profile with statistics
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")

        profile = {
            'shape': self.data.shape,
            'columns': self.data.columns.tolist(),
            'dtypes': self.data.dtypes.to_dict(),
            'missing_values': self.data.isnull().sum().to_dict(),
            'missing_percentage': (self.data.isnull().sum() / len(self.data) * 100).to_dict(),
            'duplicates': self.data.duplicated().sum(),
            'memory_usage': self.data.memory_usage(deep=True).sum() / 1024**2,  # MB
            'numeric_columns': self.data.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical_columns': self.data.select_dtypes(include=['object', 'category']).columns.tolist()
        }

        # Numeric column statistics
        if profile['numeric_columns']:
            profile['numeric_stats'] = self.data[profile['numeric_columns']].describe().to_dict()

        self.data_info = profile
        return profile

    def print_data_profile(self):
        """Print formatted data profile."""
        if not self.data_info:
            self.generate_data_profile()

        profile = self.data_info

        print("\n" + "=" * 70)
        print("DATA PROFILE")
        print("=" * 70)

        print(f"\nDataset Shape: {profile['shape'][0]} rows × {profile['shape'][1]} columns")
        print(f"Memory Usage: {profile['memory_usage']:.2f} MB")
        print(f"Duplicate Rows: {profile['duplicates']}")

        print(f"\nColumn Types:")
        print(f"  • Numeric: {len(profile['numeric_columns'])}")
        print(f"  • Categorical: {len(profile['categorical_columns'])}")

        print(f"\nMissing Values:")
        missing = {k: v for k, v in profile['missing_values'].items() if v > 0}
        if missing:
            for col, count in missing.items():
                pct = profile['missing_percentage'][col]
                print(f"  • {col}: {count} ({pct:.2f}%)")
        else:
            print("  • No missing values")

        print("\n" + "=" * 70 + "\n")

    def handle_missing_values(
        self,
        strategy: str = 'drop',
        threshold: float = 0.5,
        fill_value: Any = None
    ) -> pd.DataFrame:
        """
        Handle missing values in the dataset.

        Parameters:
        -----------
        strategy : str
            Strategy for handling missing values:
            - 'drop': drop rows with missing values
            - 'drop_cols': drop columns with missing % > threshold
            - 'mean': fill with mean (numeric only)
            - 'median': fill with median (numeric only)
            - 'mode': fill with mode
            - 'constant': fill with constant value
        threshold : float
            Threshold for dropping columns (0-1)
        fill_value : Any
            Value for constant strategy

        Returns:
        --------
        pd.DataFrame
            Data with missing values handled
        """
        if self.data is None:
            raise ValueError("No data loaded.")

        print(f"Handling missing values using strategy: {strategy}")
        initial_shape = self.data.shape

        if strategy == 'drop':
            self.data = self.data.dropna()
        elif strategy == 'drop_cols':
            missing_pct = self.data.isnull().sum() / len(self.data)
            cols_to_keep = missing_pct[missing_pct <= threshold].index
            self.data = self.data[cols_to_keep]
        elif strategy in ['mean', 'median', 'mode']:
            numeric_cols = self.data.select_dtypes(include=[np.number]).columns
            if strategy == 'mean':
                self.data[numeric_cols] = self.data[numeric_cols].fillna(
                    self.data[numeric_cols].mean()
                )
            elif strategy == 'median':
                self.data[numeric_cols] = self.data[numeric_cols].fillna(
                    self.data[numeric_cols].median()
                )
            elif strategy == 'mode':
                for col in self.data.columns:
                    if self.data[col].isnull().any():
                        mode_value = self.data[col].mode()[0] if not self.data[col].mode().empty else fill_value
                        self.data[col] = self.data[col].fillna(mode_value)
        elif strategy == 'constant':
            self.data = self.data.fillna(fill_value)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        final_shape = self.data.shape
        print(f"Shape changed: {initial_shape} → {final_shape}")

        return self.data

    def handle_outliers(
        self,
        columns: Optional[List[str]] = None,
        method: str = 'iqr',
        threshold: float = 1.5
    ) -> pd.DataFrame:
        """
        Handle outliers in numeric columns.

        Parameters:
        -----------
        columns : Optional[List[str]]
            Columns to check for outliers (default: all numeric)
        method : str
            Method for outlier detection:
            - 'iqr': Interquartile range
            - 'zscore': Z-score
        threshold : float
            Threshold for outlier detection
            - IQR: multiplier (typically 1.5)
            - Z-score: number of standard deviations (typically 3)

        Returns:
        --------
        pd.DataFrame
            Data with outliers handled
        """
        if self.data is None:
            raise ValueError("No data loaded.")

        if columns is None:
            columns = self.data.select_dtypes(include=[np.number]).columns.tolist()

        print(f"Detecting outliers using {method} method...")
        initial_len = len(self.data)

        outlier_mask = pd.Series([False] * len(self.data), index=self.data.index)

        for col in columns:
            if method == 'iqr':
                Q1 = self.data[col].quantile(0.25)
                Q3 = self.data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                outliers = (self.data[col] < lower_bound) | (self.data[col] > upper_bound)
            elif method == 'zscore':
                z_scores = np.abs((self.data[col] - self.data[col].mean()) / self.data[col].std())
                outliers = z_scores > threshold
            else:
                raise ValueError(f"Unknown method: {method}")

            outlier_mask |= outliers
            num_outliers = outliers.sum()
            if num_outliers > 0:
                print(f"  • {col}: {num_outliers} outliers detected")

        self.data = self.data[~outlier_mask]
        final_len = len(self.data)

        print(f"Removed {initial_len - final_len} rows with outliers")
        print(f"Shape after outlier removal: {self.data.shape}")

        return self.data

    def encode_categorical_variables(
        self,
        columns: Optional[List[str]] = None,
        method: str = 'onehot'
    ) -> pd.DataFrame:
        """
        Encode categorical variables.

        Parameters:
        -----------
        columns : Optional[List[str]]
            Columns to encode (default: all categorical)
        method : str
            Encoding method: 'onehot', 'label'

        Returns:
        --------
        pd.DataFrame
            Data with encoded categorical variables
        """
        if self.data is None:
            raise ValueError("No data loaded.")

        if columns is None:
            columns = self.data.select_dtypes(include=['object', 'category']).columns.tolist()

        if not columns:
            print("No categorical columns to encode")
            return self.data

        print(f"Encoding {len(columns)} categorical columns using {method} encoding...")

        if method == 'onehot':
            self.data = pd.get_dummies(self.data, columns=columns, drop_first=True)
        elif method == 'label':
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            for col in columns:
                self.data[col] = le.fit_transform(self.data[col].astype(str))
        else:
            raise ValueError(f"Unknown method: {method}")

        print(f"New shape after encoding: {self.data.shape}")

        return self.data

    def split_features_target(self, target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Split data into features and target.

        Parameters:
        -----------
        target_col : str
            Target column name

        Returns:
        --------
        Tuple[pd.DataFrame, pd.Series]
            Features and target
        """
        if self.data is None:
            raise ValueError("No data loaded.")

        if target_col not in self.data.columns:
            raise ValueError(f"Target column '{target_col}' not found in data")

        X = self.data.drop(columns=[target_col])
        y = self.data[target_col]

        print(f"Features: {X.shape[1]} columns")
        print(f"Target: {target_col}")

        return X, y

    def save_processed_data(self, filepath: str):
        """
        Save processed data to file.

        Parameters:
        -----------
        filepath : str
            Path to save data
        """
        if self.data is None:
            raise ValueError("No data to save.")

        if filepath.endswith('.csv'):
            self.data.to_csv(filepath, index=False)
        elif filepath.endswith('.parquet'):
            self.data.to_parquet(filepath, index=False)
        elif filepath.endswith('.pkl') or filepath.endswith('.pickle'):
            self.data.to_pickle(filepath)
        else:
            raise ValueError(f"Unsupported file format: {filepath}")

        print(f"Processed data saved to {filepath}")


if __name__ == "__main__":
    # Example usage
    print("Data Processing Module")
    print("Load, clean, and prepare data for ML modeling.")
    print("\nUsage example:")
    print("""
    from data_processing import DataProcessor

    # Initialize processor
    processor = DataProcessor()

    # Load data
    data = processor.load_data('data/raw/dataset.csv')

    # Generate profile
    processor.print_data_profile()

    # Handle missing values
    data = processor.handle_missing_values(strategy='median')

    # Handle outliers
    data = processor.handle_outliers(method='iqr', threshold=1.5)

    # Encode categorical variables
    data = processor.encode_categorical_variables(method='onehot')

    # Save processed data
    processor.save_processed_data('data/processed/dataset_clean.csv')
    """)
