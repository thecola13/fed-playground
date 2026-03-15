import numpy as np
import pandas as pd
from typing import Tuple, List, Optional

class DataLoader:
    """
    Handles loading data from files and preparing it for the environment.

    Supports three input modes:
    1. file_path: Load data from CSV file
    2. dataframe: Use an existing pandas DataFrame
    3. X, y: Use numpy arrays or pandas Series/DataFrames directly

    Supports two data formats:
    - Standard: Samples as rows, features as columns (default)
    - Transposed: Features as rows, samples as columns (set transpose=True)
      Used by master thesis data (e.g., Metabric.csv with genes as rows)
    """
    def __init__(
        self,
        file_path: Optional[str] = None,
        target_column: str = "target",
        feature_columns: Optional[List[str]] = None,
        dataframe: Optional[pd.DataFrame] = None,
        X: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
        transpose: bool = False,
        index_col: Optional[int] = None
    ):
        """
        Initialize DataLoader.

        Args:
            file_path: Path to CSV file
            target_column: Name of target column/row
            feature_columns: List of feature column/row names (optional)
            dataframe: Pandas DataFrame (alternative to file_path)
            X: Feature array (alternative input)
            y: Target array (alternative input)
            transpose: If True, expects features as rows, samples as columns.
                      The data will be transposed to standard format (samples × features).
            index_col: Column index to use as row names when loading CSV (default: None)
        """
        self.file_path = file_path
        self.target_column = target_column
        self.feature_columns = feature_columns
        self.dataframe = dataframe
        self.X = X
        self.y = y
        self.transpose = transpose
        self.index_col = index_col if index_col is not None else (0 if transpose else None)

        # Validate that at least one input method is provided
        input_methods = sum([
            file_path is not None,
            dataframe is not None,
            (X is not None and y is not None)
        ])

        if input_methods == 0:
            raise ValueError("Must provide either file_path, dataframe, or (X, y).")
        if input_methods > 1:
            raise ValueError("Provide only one of: file_path, dataframe, or (X, y).")
        
    def load(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Loads data and returns features and target as numpy arrays.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Features (X) and target (y) as numpy arrays
                Always in standard format: (n_samples, n_features)
        """
        # If X and y are provided directly, convert to numpy if needed
        if self.X is not None and self.y is not None:
            X = self.X if isinstance(self.X, np.ndarray) else np.array(self.X)
            y = self.y if isinstance(self.y, np.ndarray) else np.array(self.y)
            return X, y

        # Load from dataframe or file
        if self.dataframe is not None:
             df = self.dataframe.copy()
        elif self.file_path.endswith('.csv'):
            df = pd.read_csv(self.file_path, index_col=self.index_col)
        else:
            # Fallback to csv or raise error.
            # For simplicity, let's treat it as csv.
            df = pd.read_csv(self.file_path, index_col=self.index_col)

        # Handle transposed format (features as rows, samples as columns)
        if self.transpose:
            return self._load_transposed(df)
        else:
            return self._load_standard(df)

    def _load_standard(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load data in standard format (samples as rows, features as columns).

        Args:
            df: DataFrame with samples as rows

        Returns:
            Tuple of (X, y) arrays
        """
        # Select Features
        if self.feature_columns:
            X = df[self.feature_columns].values
        else:
            # Use all columns except target
            cols = [c for c in df.columns if c != self.target_column]
            X = df[cols].values

        # Select Target
        y = df[self.target_column].values

        return X, y

    def _load_transposed(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load data in transposed format (features as rows, samples as columns).
        Transposes to standard format before returning.

        This is compatible with master thesis data format where:
        - Rows are features/genes
        - Columns are samples
        - Index (first column) contains feature names

        Args:
            df: DataFrame with features as rows (index contains feature names)

        Returns:
            Tuple of (X, y) arrays in standard format (samples × features)
        """
        # Extract target row
        if self.target_column not in df.index:
            raise ValueError(f"Target '{self.target_column}' not found in data index. "
                           f"Available: {list(df.index)[:10]}...")

        y = df.loc[self.target_column].values

        # Extract feature rows
        if self.feature_columns:
            missing = [f for f in self.feature_columns if f not in df.index]
            if missing:
                raise ValueError(f"Features {missing} not found in data index. "
                               f"Available: {list(df.index)[:10]}...")
            X_df = df.loc[self.feature_columns]
        else:
            # Use all rows except target
            feature_rows = [idx for idx in df.index if idx != self.target_column]
            X_df = df.loc[feature_rows]

        # Transpose to get samples as rows, features as columns
        X = X_df.T.values

        return X, y
