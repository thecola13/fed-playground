"""Tests for fed_playground.src.aggregation."""

import numpy as np
import pytest

from fed_playground.src.aggregation import (
    GeometricMedianAggregation,
    KrumAggregation,
    MeanAggregation,
)
from fed_playground.src.encryption import AdditiveSecretSharing, NoEncryption


class TestMeanAggregation:
    def setup_method(self):
        self.strategy = MeanAggregation()
        self.scheme = NoEncryption()

    def test_mean_of_two_arrays(self):
        a = np.array([0.0, 2.0])
        b = np.array([2.0, 4.0])
        result = self.strategy.aggregate([a, b], self.scheme)
        np.testing.assert_array_almost_equal(result, np.array([1.0, 3.0]))

    def test_mean_of_identical_arrays(self):
        params = np.array([1.0, 2.0, 3.0])
        result = self.strategy.aggregate([params, params, params], self.scheme)
        np.testing.assert_array_almost_equal(result, params)

    def test_single_party(self):
        params = np.array([4.0, 5.0])
        result = self.strategy.aggregate([params], self.scheme)
        np.testing.assert_array_almost_equal(result, params)

    def test_empty_list_raises(self):
        with pytest.raises(ValueError, match="empty"):
            self.strategy.aggregate([], self.scheme)

    def test_result_is_mean_not_sum(self):
        arrays = [np.ones(3) * i for i in range(1, 6)]  # [1,1,1], [2,2,2], ..., [5,5,5]
        result = self.strategy.aggregate(arrays, self.scheme)
        expected = np.ones(3) * 3.0  # mean of 1..5
        np.testing.assert_array_almost_equal(result, expected)


def _clustered_with_outlier():
    """Five honest updates near [1,1] plus one far Byzantine outlier."""
    honest = [
        np.array([1.0, 1.0]),
        np.array([1.1, 0.9]),
        np.array([0.9, 1.1]),
        np.array([1.05, 1.0]),
        np.array([0.95, 1.05]),
    ]
    outlier = np.array([100.0, -100.0])
    return honest, outlier


class TestKrumAggregation:
    def setup_method(self):
        self.scheme = NoEncryption()

    def test_selects_honest_update_over_outlier(self):
        honest, outlier = _clustered_with_outlier()
        result = KrumAggregation(n_byzantine=1).aggregate(honest + [outlier], self.scheme)
        # Krum must pick an update inside the honest cluster, never the outlier.
        assert np.linalg.norm(result - np.array([1.0, 1.0])) < 0.5
        assert not np.allclose(result, outlier)

    def test_mean_is_dragged_by_outlier_but_krum_is_not(self):
        honest, outlier = _clustered_with_outlier()
        updates = honest + [outlier]
        mean = MeanAggregation().aggregate(updates, self.scheme)
        krum = KrumAggregation(n_byzantine=1).aggregate(updates, self.scheme)
        assert np.linalg.norm(mean - np.array([1.0, 1.0])) > 10.0  # mean corrupted
        assert np.linalg.norm(krum - np.array([1.0, 1.0])) < 0.5   # krum robust

    def test_multi_krum_averages_selected(self):
        honest, outlier = _clustered_with_outlier()
        result = KrumAggregation(n_byzantine=1, n_selected=3).aggregate(
            honest + [outlier], self.scheme
        )
        assert np.linalg.norm(result - np.array([1.0, 1.0])) < 0.5

    def test_few_parties_fall_back_to_mean(self):
        a, b = np.array([1.0, 2.0]), np.array([3.0, 4.0])
        result = KrumAggregation().aggregate([a, b], self.scheme)
        np.testing.assert_array_almost_equal(result, np.array([2.0, 3.0]))

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            KrumAggregation().aggregate([], self.scheme)

    def test_rejects_masking_scheme(self):
        with pytest.raises(ValueError, match="masks"):
            KrumAggregation().aggregate(
                [np.ones(2), np.ones(2), np.ones(2)], AdditiveSecretSharing(seed=0)
            )


class TestGeometricMedianAggregation:
    def setup_method(self):
        self.scheme = NoEncryption()

    def test_robust_to_outlier(self):
        honest, outlier = _clustered_with_outlier()
        result = GeometricMedianAggregation().aggregate(honest + [outlier], self.scheme)
        assert np.linalg.norm(result - np.array([1.0, 1.0])) < 0.5

    def test_identical_updates_return_that_point(self):
        p = np.array([2.0, -3.0, 5.0])
        result = GeometricMedianAggregation().aggregate([p, p, p], self.scheme)
        np.testing.assert_array_almost_equal(result, p)

    def test_matches_mean_for_symmetric_cloud(self):
        # For a symmetric configuration the geometric median equals the centroid.
        pts = [np.array([1.0, 0.0]), np.array([-1.0, 0.0]),
               np.array([0.0, 1.0]), np.array([0.0, -1.0])]
        result = GeometricMedianAggregation().aggregate(pts, self.scheme)
        np.testing.assert_array_almost_equal(result, np.zeros(2), decimal=4)

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            GeometricMedianAggregation().aggregate([], self.scheme)

    def test_rejects_masking_scheme(self):
        with pytest.raises(ValueError, match="masks"):
            GeometricMedianAggregation().aggregate(
                [np.ones(2), np.ones(2)], AdditiveSecretSharing(seed=0)
            )
