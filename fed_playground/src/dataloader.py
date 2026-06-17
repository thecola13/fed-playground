"""Data loading utilities for the federated learning environment.

:class:`DataLoader` provides a unified interface for supplying data to
:class:`~fed_playground.src.environment.Environment` from three different
sources: a CSV file path, an existing pandas DataFrame, or pre-split numpy /
pandas arrays.
"""

import numpy as np
import pandas as pd


class DataLoader:
    """Load and prepare data for a federated learning simulation.

    Exactly one of *file_path*, *dataframe*, or the ``(X, y)`` pair must be
    supplied; providing more than one (or none) raises :exc:`ValueError`.

    Supports two data layouts:

    * **Standard** (default): rows are samples, columns are features.
    * **Transposed** (``transpose=True``): rows are features/genes, columns are
      samples — typical of bioinformatics datasets such as Metabric.

    Args:
        file_path: Path to a CSV file.  Mutually exclusive with *dataframe*
            and ``(X, y)``.
        target_column: Name of the column (standard) or row index (transposed)
            that contains the target variable.  Default is ``"target"``.
        feature_columns: Explicit list of feature column/row names.  When
            ``None`` all columns/rows except *target_column* are used.
        dataframe: Pre-loaded pandas DataFrame.  Mutually exclusive with
            *file_path* and ``(X, y)``.
        X: Feature array (numpy array or pandas DataFrame/Series).  Must be
            paired with *y*.
        y: Target array (numpy array or pandas Series).  Must be paired with
            *X*.
        transpose: When ``True`` the loaded DataFrame is transposed before
            feature / target extraction.
        index_col: Column index to use as the row index when reading a CSV.
            Defaults to ``0`` when *transpose* is ``True``, ``None`` otherwise.

    Raises:
        ValueError: If zero or more than one input source is provided.
    """

    def __init__(
        self,
        file_path: str | None = None,
        target_column: str = "target",
        feature_columns: list[str] | None = None,
        dataframe: pd.DataFrame | None = None,
        X: np.ndarray | None = None,
        y: np.ndarray | None = None,
        transpose: bool = False,
        index_col: int | None = None,
    ) -> None:
        self.file_path = file_path
        self.target_column = target_column
        self.feature_columns = feature_columns
        self.dataframe = dataframe
        self.X = X
        self.y = y
        self.transpose = transpose
        self.index_col = (
            index_col if index_col is not None else (0 if transpose else None)
        )

        n_sources = sum(
            [
                file_path is not None,
                dataframe is not None,
                (X is not None and y is not None),
            ]
        )
        if n_sources != 1:
            raise ValueError("Provide exactly one of: file_path, dataframe, or (X, y).")

    def load(self) -> tuple[np.ndarray, np.ndarray]:
        """Load data and return feature matrix and target vector.

        Returns:
            Tuple ``(X, y)`` where *X* has shape ``(n_samples, n_features)``
            and *y* has shape ``(n_samples,)``.  Both are numpy arrays in
            standard (samples × features) layout regardless of *transpose*.

        Raises:
            ValueError: If the target column/row is missing from the data.
            ValueError: If any requested feature columns/rows are missing.
        """
        # Passthrough for pre-split arrays
        if self.X is not None and self.y is not None:
            X = self.X if isinstance(self.X, np.ndarray) else np.array(self.X)
            y = self.y if isinstance(self.y, np.ndarray) else np.array(self.y)
            return X, y

        # Load from DataFrame or file
        if self.dataframe is not None:
            df = self.dataframe.copy()
        elif self.file_path is not None:
            df = pd.read_csv(self.file_path, index_col=self.index_col)
        else:
            # Unreachable: init validation ensures at least one source exists.
            raise RuntimeError("No data source available.")  # pragma: no cover

        if self.transpose:
            return self._load_transposed(df)
        return self._load_standard(df)

    def _load_standard(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Extract features and target from a standard-layout DataFrame.

        Args:
            df: DataFrame with samples as rows and features as columns.

        Returns:
            Tuple ``(X, y)`` as numpy arrays.

        Raises:
            KeyError: If *target_column* or any *feature_columns* are absent.
        """
        if self.feature_columns:
            X = df[self.feature_columns].values
        else:
            cols = [c for c in df.columns if c != self.target_column]
            X = df[cols].values

        y = df[self.target_column].values
        return X, y

    def _load_transposed(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Extract features and target from a transposed-layout DataFrame.

        Rows are features; columns are samples.  The DataFrame is transposed
        internally so the returned arrays are in standard layout.

        Args:
            df: DataFrame with features as rows (index holds feature names).

        Returns:
            Tuple ``(X, y)`` in standard ``(n_samples, n_features)`` layout.

        Raises:
            ValueError: If *target_column* is not in ``df.index``.
            ValueError: If any of *feature_columns* are not in ``df.index``.
        """
        if self.target_column not in df.index:
            available = list(df.index)[:10]
            raise ValueError(
                f"Target '{self.target_column}' not found in data index. "
                f"First 10 available: {available}"
            )

        y = df.loc[self.target_column].values

        if self.feature_columns:
            missing = [f for f in self.feature_columns if f not in df.index]
            if missing:
                available = list(df.index)[:10]
                raise ValueError(
                    f"Feature(s) {missing} not found in data index. "
                    f"First 10 available: {available}"
                )
            X_df = df.loc[self.feature_columns]
        else:
            feature_rows = [idx for idx in df.index if idx != self.target_column]
            X_df = df.loc[feature_rows]

        X = X_df.T.values
        return X, y


def load_dataset(kind: str = "synthetic", **opts: object) -> "DataLoader":
    """Build a :class:`DataLoader` for a named dataset (the benchmark data layer).

    Args:
        kind: one of ``synthetic`` | ``sklearn`` | ``openml`` | ``csv``.
        **opts: per-kind options —
            synthetic: ``n_samples``, ``n_features``, ``seed``;
            sklearn: ``name`` in {``breast_cancer``, ``diabetes``} (offline);
            openml: MNIST via network (no opts);
            csv: ``path`` (and optional ``target``).

    Returns:
        A ready-to-use ``DataLoader``.

    Raises:
        ValueError: on an unknown *kind*.
        ImportError: if ``sklearn`` is needed but not installed (``[examples]`` extra).
    """
    from .utils_data import generate_linear_data  # local: avoids import cycle

    if kind == "synthetic":
        X, y = generate_linear_data(
            int(opts.get("n_samples", 500)),  # type: ignore[arg-type]
            int(opts.get("n_features", 4)),  # type: ignore[arg-type]
            random_seed=int(opts.get("seed", 42)),  # type: ignore[arg-type]
        )
        return DataLoader(X=X, y=y)
    if kind == "csv":
        return DataLoader(
            file_path=str(opts["path"]),
            target_column=str(opts.get("target", "target")),
        )
    if kind == "sklearn":
        try:
            from sklearn import datasets as skd
        except ImportError as exc:  # pragma: no cover - env-dependent
            raise ImportError(
                "sklearn datasets need the extra: `uv sync --extra examples`."
            ) from exc
        loaders = {
            "breast_cancer": skd.load_breast_cancer,
            "diabetes": skd.load_diabetes,
        }
        bunch = loaders[str(opts.get("name", "diabetes"))]()
        return DataLoader(X=bunch.data, y=bunch.target)
    if kind == "openml":  # pragma: no cover - network
        from sklearn.datasets import fetch_openml

        mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
        return DataLoader(
            X=mnist.data.astype("float32") / 255.0, y=mnist.target.astype(int)
        )
    raise ValueError(f"unknown dataset kind {kind!r}")
