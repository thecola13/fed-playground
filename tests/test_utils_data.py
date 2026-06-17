"""Tests for fed_playground.src.utils_data."""

import numpy as np
import pytest

from fed_playground.src.utils_data import generate_linear_data, split_data


class TestGenerateLinearData:
    def test_output_shapes(self):
        X, y = generate_linear_data(n_samples=100, n_features=5)
        assert X.shape == (100, 5)
        assert y.shape == (100,)

    def test_reproducibility(self):
        X1, y1 = generate_linear_data(100, 5, random_seed=0)
        X2, y2 = generate_linear_data(100, 5, random_seed=0)
        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)

    def test_different_seeds_differ(self):
        X1, _ = generate_linear_data(100, 5, random_seed=1)
        X2, _ = generate_linear_data(100, 5, random_seed=2)
        assert not np.allclose(X1, X2)

    def test_noise_zero(self):
        """With zero noise all variance comes from the linear component."""
        _X, y = generate_linear_data(50, 3, noise=0.0, random_seed=7)
        assert np.isfinite(y).all()


class TestSplitData:
    def setup_method(self):
        X, y = generate_linear_data(100, 4, random_seed=42)
        self.X, self.y = X, y

    def test_number_of_splits(self):
        splits = split_data(self.X, self.y, n_parties=4)
        assert len(splits) == 4

    def test_total_samples_preserved(self):
        splits = split_data(self.X, self.y, n_parties=3)
        total = sum(Xi.shape[0] for Xi, _ in splits)
        assert total == self.X.shape[0]

    def test_feature_dim_preserved(self):
        splits = split_data(self.X, self.y, n_parties=5)
        for Xi, yi in splits:
            assert Xi.shape[1] == self.X.shape[1]
            assert Xi.shape[0] == yi.shape[0]

    def test_invalid_n_parties(self):
        with pytest.raises(ValueError, match="n_parties"):
            split_data(self.X, self.y, n_parties=0)

    def test_reproducibility(self):
        s1 = split_data(self.X, self.y, n_parties=3, random_seed=10)
        s2 = split_data(self.X, self.y, n_parties=3, random_seed=10)
        for (X1, y1), (X2, y2) in zip(s1, s2, strict=True):
            np.testing.assert_array_equal(X1, X2)
            np.testing.assert_array_equal(y1, y2)
