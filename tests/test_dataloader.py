"""Tests for fed_playground.src.dataloader.

Includes a regression test for the previously broken 'elif' branch that made
the dataframe path unreachable when file_path was also absent.
"""

import numpy as np
import pandas as pd
import pytest

from fed_playground.src.dataloader import DataLoader


def _make_df(n=50, n_features=3, seed=0):
    rng = np.random.default_rng(seed)
    data = {f"f{i}": rng.standard_normal(n) for i in range(n_features)}
    data["target"] = rng.standard_normal(n)
    return pd.DataFrame(data)


class TestDataLoaderValidation:
    def test_no_source_raises(self):
        with pytest.raises(ValueError, match="exactly one"):
            DataLoader()

    def test_multiple_sources_raises(self):
        df = _make_df()
        X = np.zeros((10, 3))
        y = np.zeros(10)
        with pytest.raises(ValueError, match="exactly one"):
            DataLoader(dataframe=df, X=X, y=y)

    def test_only_x_without_y_raises(self):
        """Providing X without y counts as zero valid sources."""
        with pytest.raises(ValueError, match="exactly one"):
            DataLoader(X=np.zeros((10, 3)))


class TestDataLoaderFromArrays:
    def test_numpy_passthrough(self):
        X = np.random.randn(20, 4)
        y = np.random.randn(20)
        loader = DataLoader(X=X, y=y)
        X_out, y_out = loader.load()
        np.testing.assert_array_equal(X_out, X)
        np.testing.assert_array_equal(y_out, y)

    def test_pandas_converted_to_numpy(self):
        df = _make_df(30, 3)
        X_pd = df[["f0", "f1", "f2"]]
        y_pd = df["target"]
        loader = DataLoader(X=X_pd, y=y_pd)
        X_out, y_out = loader.load()
        assert isinstance(X_out, np.ndarray)
        assert isinstance(y_out, np.ndarray)
        assert X_out.shape == (30, 3)


class TestDataLoaderFromDataFrame:
    """Regression tests for the dataframe branch (previously broken elif)."""

    def test_dataframe_loads_correctly(self):
        df = _make_df(40, 3)
        loader = DataLoader(dataframe=df, target_column="target")
        X, y = loader.load()
        assert X.shape == (40, 3)
        assert y.shape == (40,)

    def test_dataframe_feature_selection(self):
        df = _make_df(20, 4)
        loader = DataLoader(
            dataframe=df,
            target_column="target",
            feature_columns=["f0", "f2"],
        )
        X, y = loader.load()
        assert X.shape == (20, 2)
        assert y.shape == (20,)

    def test_dataframe_missing_target_raises(self):
        df = _make_df(20, 2)
        loader = DataLoader(dataframe=df, target_column="nonexistent")
        with pytest.raises(KeyError):
            loader.load()


class TestDataLoaderFromCSV:
    def test_csv_loads_correctly(self, tmp_path):
        df = _make_df(30, 3)
        csv_path = str(tmp_path / "data.csv")
        df.to_csv(csv_path, index=False)

        loader = DataLoader(file_path=csv_path, target_column="target")
        X, y = loader.load()
        assert X.shape == (30, 3)
        assert y.shape == (30,)

    def test_csv_missing_target_raises(self, tmp_path):
        df = _make_df(10, 2)
        csv_path = str(tmp_path / "data.csv")
        df.to_csv(csv_path, index=False)

        loader = DataLoader(file_path=csv_path, target_column="missing_col")
        with pytest.raises(KeyError):
            loader.load()


class TestDataLoaderTransposed:
    def _make_transposed_df(self):
        """Build a transposed-layout DataFrame (features as rows)."""
        rng = np.random.default_rng(0)
        n_samples = 20
        data = {f"sample_{i}": rng.standard_normal(4) for i in range(n_samples)}
        df = pd.DataFrame(data, index=["gene_a", "gene_b", "gene_c", "target"])
        return df

    def test_transposed_load_shape(self):
        df = self._make_transposed_df()
        loader = DataLoader(dataframe=df, target_column="target", transpose=True)
        X, y = loader.load()
        assert X.shape == (20, 3)  # 3 feature rows, 20 samples
        assert y.shape == (20,)

    def test_transposed_missing_target_raises(self):
        df = self._make_transposed_df()
        loader = DataLoader(dataframe=df, target_column="no_such_row", transpose=True)
        with pytest.raises(ValueError, match="not found in data index"):
            loader.load()
