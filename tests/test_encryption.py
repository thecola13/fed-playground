"""Tests for fed_playground.src.encryption."""

import numpy as np
import pytest

from fed_playground.src.aggregation import MeanAggregation, MedianAggregation
from fed_playground.src.encryption import (
    AdditiveSecretSharing,
    LaplaceDPEncryption,
    NoEncryption,
    PairwiseMaskingEncryption,
)


class TestNoEncryption:
    def setup_method(self):
        self.scheme = NoEncryption()
        self.params = np.array([1.0, 2.0, 3.0])

    def test_encrypt_returns_same_array(self):
        result = self.scheme.encrypt(self.params)
        np.testing.assert_array_equal(result, self.params)

    def test_decrypt_returns_same_array(self):
        result = self.scheme.decrypt(self.params)
        np.testing.assert_array_equal(result, self.params)

    def test_roundtrip(self):
        encrypted = self.scheme.encrypt(self.params)
        decrypted = self.scheme.decrypt(encrypted)
        np.testing.assert_array_equal(decrypted, self.params)

    def test_aggregate_sum(self):
        a = np.array([1.0, 2.0])
        b = np.array([3.0, 4.0])
        result = self.scheme.aggregate([a, b])
        np.testing.assert_array_equal(result, np.array([4.0, 6.0]))

    def test_aggregate_single(self):
        a = np.array([5.0, 6.0])
        result = self.scheme.aggregate([a])
        np.testing.assert_array_equal(result, a)

    def test_aggregate_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            self.scheme.aggregate([])


class TestAdditiveSecretSharing:
    def setup_method(self):
        self.scheme = AdditiveSecretSharing(seed=0)
        self.updates = [
            np.array([1.0, 2.0, 3.0]),
            np.array([4.0, 5.0, 6.0]),
            np.array([-1.0, 0.0, 7.0]),
        ]

    def test_public_share_hides_the_update(self):
        # The emitted share must not equal the plaintext update.
        x = self.updates[0]
        share = self.scheme.encrypt(x)
        assert not np.allclose(share, x)

    def test_aggregate_reconstructs_exact_sum(self):
        shares = [self.scheme.encrypt(x) for x in self.updates]
        result = self.scheme.aggregate(shares)
        np.testing.assert_allclose(result, sum(self.updates))

    def test_mean_aggregation_recovers_true_mean(self):
        shares = [self.scheme.encrypt(x) for x in self.updates]
        result = MeanAggregation().aggregate(shares, self.scheme)
        np.testing.assert_allclose(result, sum(self.updates) / len(self.updates))

    def test_ledger_resets_between_rounds(self):
        # Two independent rounds must each reconstruct their own sum.
        for _ in range(2):
            shares = [self.scheme.encrypt(x) for x in self.updates]
            np.testing.assert_allclose(self.scheme.aggregate(shares), sum(self.updates))

    def test_n_shares_above_two_still_reconstructs(self):
        scheme = AdditiveSecretSharing(n_shares=4, seed=1)
        shares = [scheme.encrypt(x) for x in self.updates]
        np.testing.assert_allclose(scheme.aggregate(shares), sum(self.updates))

    def test_median_aggregation_refuses_masked_shares(self):
        shares = [self.scheme.encrypt(x) for x in self.updates]
        with pytest.raises(ValueError, match="order"):
            MedianAggregation().aggregate(shares, self.scheme)

    def test_n_shares_below_two_raises(self):
        with pytest.raises(ValueError, match="at least 2"):
            AdditiveSecretSharing(n_shares=1)


class TestLaplaceDPEncryption:
    def test_adds_noise(self):
        scheme = LaplaceDPEncryption(epsilon=1.0, seed=0)
        x = np.zeros(5)
        noisy = scheme.encrypt(x)
        assert not np.allclose(noisy, x)

    def test_scale_is_sensitivity_over_epsilon(self):
        scheme = LaplaceDPEncryption(epsilon=0.5, sensitivity=2.0)
        assert scheme.scale == pytest.approx(4.0)

    def test_smaller_epsilon_means_more_noise(self):
        # Empirical noise std scales as sqrt(2)*b = sqrt(2)*sensitivity/epsilon.
        x = np.zeros(20000)
        tight = LaplaceDPEncryption(epsilon=4.0, seed=1).encrypt(x)
        loose = LaplaceDPEncryption(epsilon=0.5, seed=1).encrypt(x)
        assert loose.std() > tight.std()

    def test_clip_bounds_l1_sensitivity(self):
        scheme = LaplaceDPEncryption(epsilon=1e12, sensitivity=1.0, clip=True, seed=0)
        # Huge epsilon -> negligible noise, so output ~ the clipped input.
        out = scheme.encrypt(np.array([10.0, -10.0, 10.0]))
        assert np.sum(np.abs(out)) == pytest.approx(1.0, abs=1e-3)

    def test_aggregate_sums(self):
        scheme = LaplaceDPEncryption(seed=0)
        a, b = np.array([1.0, 2.0]), np.array([3.0, 4.0])
        np.testing.assert_array_equal(scheme.aggregate([a, b]), np.array([4.0, 6.0]))

    def test_invalid_epsilon_raises(self):
        with pytest.raises(ValueError, match="epsilon"):
            LaplaceDPEncryption(epsilon=0.0)


class TestPairwiseMaskingEncryption:
    def setup_method(self):
        self.n = 4
        self.scheme = PairwiseMaskingEncryption(n_parties=self.n, seed=0)
        self.updates = [np.array([float(i), float(-i), 1.0]) for i in range(self.n)]

    def test_individual_shares_are_masked(self):
        share = self.scheme.encrypt(self.updates[0])
        assert not np.allclose(share, self.updates[0])

    def test_masks_cancel_to_exact_sum(self):
        shares = [self.scheme.encrypt(x) for x in self.updates]
        np.testing.assert_allclose(self.scheme.aggregate(shares), sum(self.updates))

    def test_mean_aggregation_recovers_true_mean(self):
        shares = [self.scheme.encrypt(x) for x in self.updates]
        result = MeanAggregation().aggregate(shares, self.scheme)
        np.testing.assert_allclose(result, sum(self.updates) / self.n)

    def test_independent_rounds_each_reconstruct(self):
        for _ in range(3):
            shares = [self.scheme.encrypt(x) for x in self.updates]
            np.testing.assert_allclose(self.scheme.aggregate(shares), sum(self.updates))

    def test_party_masks_sum_to_zero(self):
        masks = [self.scheme._party_mask(0, i, (3,)) for i in range(self.n)]
        np.testing.assert_allclose(sum(masks), np.zeros(3), atol=1e-9)

    def test_rejected_by_median_aggregation(self):
        shares = [self.scheme.encrypt(x) for x in self.updates]
        with pytest.raises(ValueError, match="masks"):
            MedianAggregation().aggregate(shares, self.scheme)

    def test_invalid_n_parties_raises(self):
        with pytest.raises(ValueError, match="n_parties"):
            PairwiseMaskingEncryption(n_parties=0)
