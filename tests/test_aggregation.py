"""Tests for fed_playground.src.aggregation."""

import numpy as np
import pytest

from fed_playground.src.aggregation import (
    BulyanAggregation,
    CenteredClippingAggregation,
    GeometricMedianAggregation,
    KrumAggregation,
    MeanAggregation,
    MedianOfMeansAggregation,
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
        result = KrumAggregation(n_byzantine=1).aggregate(
            [*honest, outlier], self.scheme
        )
        # Krum must pick an update inside the honest cluster, never the outlier.
        assert np.linalg.norm(result - np.array([1.0, 1.0])) < 0.5
        assert not np.allclose(result, outlier)

    def test_mean_is_dragged_by_outlier_but_krum_is_not(self):
        honest, outlier = _clustered_with_outlier()
        updates = [*honest, outlier]
        mean = MeanAggregation().aggregate(updates, self.scheme)
        krum = KrumAggregation(n_byzantine=1).aggregate(updates, self.scheme)
        assert np.linalg.norm(mean - np.array([1.0, 1.0])) > 10.0  # mean corrupted
        assert np.linalg.norm(krum - np.array([1.0, 1.0])) < 0.5  # krum robust

    def test_multi_krum_averages_selected(self):
        honest, outlier = _clustered_with_outlier()
        result = KrumAggregation(n_byzantine=1, n_selected=3).aggregate(
            [*honest, outlier], self.scheme
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
        result = GeometricMedianAggregation().aggregate([*honest, outlier], self.scheme)
        assert np.linalg.norm(result - np.array([1.0, 1.0])) < 0.5

    def test_identical_updates_return_that_point(self):
        p = np.array([2.0, -3.0, 5.0])
        result = GeometricMedianAggregation().aggregate([p, p, p], self.scheme)
        np.testing.assert_array_almost_equal(result, p)

    def test_matches_mean_for_symmetric_cloud(self):
        # For a symmetric configuration the geometric median equals the centroid.
        pts = [
            np.array([1.0, 0.0]),
            np.array([-1.0, 0.0]),
            np.array([0.0, 1.0]),
            np.array([0.0, -1.0]),
        ]
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


class TestBulyanAggregation:
    def setup_method(self):
        self.scheme = NoEncryption()

    def test_robust_to_single_outlier(self):
        honest, outlier = _clustered_with_outlier()
        result = BulyanAggregation(n_byzantine=1).aggregate(
            [*honest, outlier], self.scheme
        )
        assert np.linalg.norm(result - np.array([1.0, 1.0])) < 0.5

    def test_robust_to_two_outliers(self):
        # 9 honest near [2,2] + 2 Byzantine; n=11 >= 4f+3 with f=2.
        rng = np.random.default_rng(0)
        honest = [np.array([2.0, 2.0]) + 0.1 * rng.standard_normal(2) for _ in range(9)]
        byz = [np.array([200.0, -200.0]), np.array([-150.0, 300.0])]
        result = BulyanAggregation(n_byzantine=2).aggregate(honest + byz, self.scheme)
        assert np.linalg.norm(result - np.array([2.0, 2.0])) < 0.5

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            BulyanAggregation().aggregate([], self.scheme)

    def test_rejects_masking_scheme(self):
        with pytest.raises(ValueError, match="masks"):
            BulyanAggregation().aggregate(
                [np.ones(2)] * 4, AdditiveSecretSharing(seed=0)
            )


class TestMedianOfMeansAggregation:
    def setup_method(self):
        self.scheme = NoEncryption()

    def test_robust_to_outlier(self):
        honest, outlier = _clustered_with_outlier()
        result = MedianOfMeansAggregation(n_buckets=5).aggregate(
            [*honest, outlier], self.scheme
        )
        assert np.linalg.norm(result - np.array([1.0, 1.0])) < 0.6

    def test_identical_updates_return_that_point(self):
        p = np.array([3.0, -1.0, 4.0])
        result = MedianOfMeansAggregation(n_buckets=3).aggregate(
            [p, p, p, p], self.scheme
        )
        np.testing.assert_array_almost_equal(result, p)

    def test_buckets_clamped_to_party_count(self):
        a, b = np.array([1.0, 1.0]), np.array([3.0, 3.0])
        # n_buckets=10 but only 2 parties -> 2 buckets -> median of the two means.
        result = MedianOfMeansAggregation(n_buckets=10).aggregate([a, b], self.scheme)
        np.testing.assert_array_almost_equal(result, np.array([2.0, 2.0]))

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            MedianOfMeansAggregation().aggregate([], self.scheme)

    def test_rejects_masking_scheme(self):
        with pytest.raises(ValueError, match="masks"):
            MedianOfMeansAggregation().aggregate(
                [np.ones(2), np.ones(2)], AdditiveSecretSharing(seed=0)
            )


class TestCenteredClippingAggregation:
    def setup_method(self):
        self.scheme = NoEncryption()

    def test_robust_to_outlier(self):
        honest, outlier = _clustered_with_outlier()
        result = CenteredClippingAggregation(clip_radius=1.0, n_iters=5).aggregate(
            [*honest, outlier], self.scheme
        )
        assert np.linalg.norm(result - np.array([1.0, 1.0])) < 0.5

    def test_converges_to_mean_on_clean_data(self):
        # With no outliers and a generous radius, clipping is inactive and the
        # iteration converges to the arithmetic mean (distinguishes it from median).
        rng = np.random.default_rng(0)
        pts = [rng.standard_normal(3) for _ in range(8)]
        result = CenteredClippingAggregation(clip_radius=100.0, n_iters=20).aggregate(
            pts, self.scheme
        )
        np.testing.assert_allclose(result, np.mean(pts, axis=0), atol=1e-6)

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            CenteredClippingAggregation().aggregate([], self.scheme)

    def test_invalid_params_raise(self):
        with pytest.raises(ValueError, match="clip_radius"):
            CenteredClippingAggregation(clip_radius=0.0)
        with pytest.raises(ValueError, match="n_iters"):
            CenteredClippingAggregation(n_iters=0)

    def test_rejects_masking_scheme(self):
        with pytest.raises(ValueError, match="masks"):
            CenteredClippingAggregation().aggregate(
                [np.ones(2), np.ones(2)], AdditiveSecretSharing(seed=0)
            )
