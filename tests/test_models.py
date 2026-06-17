"""Tests for fed_playground.src.models."""

import numpy as np
import pytest

from fed_playground.src.models import (
    ClosedFormLinearRegressionModel,
    ElasticNetRegressionModel,
    HuberRegressionModel,
    LassoRegressionModel,
    LinearRegressionModel,
    PoissonRegressionModel,
    SVMModel,
)
from fed_playground.src.utils_data import generate_linear_data


def _linearly_separable(n=200, d=4, seed=0):
    """Two Gaussian blobs separable by a hyperplane; labels in {0, 1}."""
    rng = np.random.default_rng(seed)
    half = n // 2
    pos = rng.standard_normal((half, d)) + 2.5
    neg = rng.standard_normal((half, d)) - 2.5
    X = np.vstack([pos, neg])
    y = np.concatenate([np.ones(half), np.zeros(half)])
    perm = rng.permutation(n)
    return X[perm], y[perm]


class TestSVMModel:
    def test_separates_linearly_separable_data(self):
        X, y = _linearly_separable(seed=1)
        model = SVMModel(input_dim=X.shape[1], lambda_reg=0.01, epochs=30)
        model.train(X, y)
        assert model.evaluate(X, y) > 0.95

    def test_predictions_are_binary(self):
        X, y = _linearly_separable(seed=2)
        model = SVMModel(input_dim=X.shape[1])
        model.train(X, y)
        preds = model.predict(X)
        assert set(np.unique(preds)).issubset({0.0, 1.0})

    def test_parameter_roundtrip(self):
        X, y = _linearly_separable(seed=3)
        model = SVMModel(input_dim=X.shape[1])
        model.train(X, y)
        params = model.get_parameters()
        assert params.shape == (X.shape[1] + 1,)
        clone = SVMModel(input_dim=X.shape[1])
        clone.set_parameters(params)
        np.testing.assert_array_equal(clone.predict(X), model.predict(X))

    def test_empty_train_no_crash(self):
        SVMModel(input_dim=3).train(np.zeros((0, 3)), np.zeros(0))

    def test_invalid_lambda_raises(self):
        with pytest.raises(ValueError, match="lambda_reg"):
            SVMModel(input_dim=3, lambda_reg=0.0)


class TestLassoRegressionModel:
    def test_recovers_ols_when_alpha_near_zero(self):
        X, y = generate_linear_data(
            n_samples=200, n_features=4, noise=0.01, random_seed=0
        )
        lasso = LassoRegressionModel(input_dim=4, alpha=1e-6)
        lasso.train(X, y)
        ols = ClosedFormLinearRegressionModel(input_dim=4)
        ols.train(X, y)
        # Predictions should agree closely with ordinary least squares.
        np.testing.assert_allclose(lasso.predict(X), ols.predict(X), atol=0.05)

    def test_induces_sparsity_on_irrelevant_features(self):
        # Only the first two of six features drive the target.
        rng = np.random.default_rng(0)
        X = rng.standard_normal((300, 6))
        y = 3.0 * X[:, 0] - 2.0 * X[:, 1] + 0.01 * rng.standard_normal(300)
        lasso = LassoRegressionModel(input_dim=6, alpha=0.3)
        lasso.train(X, y)
        relevant = np.abs(lasso.weights[:2])
        irrelevant = np.abs(lasso.weights[2:])
        assert np.all(relevant > 0.5)  # signal kept
        assert np.all(irrelevant < 1e-8)  # noise features driven to exact zero

    def test_large_alpha_zeros_all_weights(self):
        X, y = generate_linear_data(n_samples=100, n_features=4, random_seed=1)
        lasso = LassoRegressionModel(input_dim=4, alpha=1e6)
        lasso.train(X, y)
        np.testing.assert_array_almost_equal(lasso.weights, np.zeros(4))
        # With all weights zero the prediction collapses to the intercept ≈ mean(y).
        np.testing.assert_allclose(
            lasso.predict(X), np.full(len(y), np.mean(y)), atol=1e-6
        )

    def test_empty_train_no_crash(self):
        LassoRegressionModel(input_dim=3).train(np.zeros((0, 3)), np.zeros(0))


class TestElasticNetRegressionModel:
    def _sparse_problem(self, seed=0):
        rng = np.random.default_rng(seed)
        X = rng.standard_normal((300, 6))
        y = 3.0 * X[:, 0] - 2.0 * X[:, 1] + 0.01 * rng.standard_normal(300)
        return X, y

    def test_recovers_ols_when_alpha_near_zero(self):
        X, y = generate_linear_data(
            n_samples=200, n_features=4, noise=0.01, random_seed=0
        )
        en = ElasticNetRegressionModel(input_dim=4, alpha=1e-6, l1_ratio=0.5)
        en.train(X, y)
        ols = ClosedFormLinearRegressionModel(input_dim=4)
        ols.train(X, y)
        np.testing.assert_allclose(en.predict(X), ols.predict(X), atol=0.05)

    def test_pure_l1_induces_exact_sparsity(self):
        X, y = self._sparse_problem()
        en = ElasticNetRegressionModel(input_dim=6, alpha=0.3, l1_ratio=1.0)
        en.train(X, y)
        assert np.all(np.abs(en.weights[2:]) < 1e-8)  # irrelevant features zeroed

    def test_pure_l2_shrinks_without_exact_zeros(self):
        X, y = self._sparse_problem()
        en = ElasticNetRegressionModel(input_dim=6, alpha=0.5, l1_ratio=0.0)
        en.train(X, y)
        # Ridge-like: irrelevant weights are small but generically not exactly 0.
        assert np.all(np.abs(en.weights[2:]) < 0.2)
        assert np.any(np.abs(en.weights[2:]) > 0.0)

    def test_invalid_l1_ratio_raises(self):
        with pytest.raises(ValueError, match="l1_ratio"):
            ElasticNetRegressionModel(input_dim=3, l1_ratio=1.5)


class TestPoissonRegressionModel:
    def _count_data(self, n=400, d=3, seed=0):
        rng = np.random.default_rng(seed)
        X = 0.5 * rng.standard_normal((n, d))
        true_w = np.array([0.8, -0.5, 0.3])[:d]
        rate = np.exp(X @ true_w + 0.2)
        y = rng.poisson(rate).astype(float)
        return X, y

    def test_predictions_are_non_negative(self):
        X, y = self._count_data()
        model = PoissonRegressionModel(input_dim=X.shape[1], epochs=100)
        model.train(X, y)
        assert np.all(model.predict(X) >= 0.0)

    def test_training_reduces_deviance(self):
        X, y = self._count_data()
        model = PoissonRegressionModel(input_dim=X.shape[1], epochs=200)
        before = model.evaluate(X, y)
        model.train(X, y)
        after = model.evaluate(X, y)
        assert after < before

    def test_recovers_rate_on_synthetic_counts(self):
        X, y = self._count_data(n=2000, seed=2)
        model = PoissonRegressionModel(
            input_dim=X.shape[1], learning_rate=0.05, epochs=500
        )
        model.train(X, y)
        # Fitted mean rate should track the empirical mean count.
        assert abs(model.predict(X).mean() - y.mean()) < 0.1

    def test_parameter_roundtrip(self):
        X, y = self._count_data()
        model = PoissonRegressionModel(input_dim=X.shape[1])
        model.train(X, y)
        clone = PoissonRegressionModel(input_dim=X.shape[1])
        clone.set_parameters(model.get_parameters())
        np.testing.assert_array_almost_equal(clone.predict(X), model.predict(X))

    def test_empty_train_no_crash(self):
        PoissonRegressionModel(input_dim=3).train(np.zeros((0, 3)), np.zeros(0))


class TestHuberRegressionModel:
    def _data_with_outliers(self, seed=0):
        X, y = generate_linear_data(
            n_samples=200, n_features=3, noise=0.05, random_seed=seed
        )
        y_corrupt = y.copy()
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(y), size=20, replace=False)  # 10% gross outliers
        y_corrupt[idx] += 100.0
        return X, y, y_corrupt

    def test_more_robust_than_ols_under_target_outliers(self):
        X, y_clean, y_corrupt = self._data_with_outliers()
        huber = HuberRegressionModel(
            input_dim=3, delta=1.0, learning_rate=0.05, epochs=300
        )
        huber.train(X, y_corrupt)
        ols = ClosedFormLinearRegressionModel(input_dim=3)
        ols.train(X, y_corrupt)
        # Error measured against the CLEAN targets: Huber should track them better.
        huber_err = np.mean((huber.predict(X) - y_clean) ** 2)
        ols_err = np.mean((ols.predict(X) - y_clean) ** 2)
        assert huber_err < ols_err

    def test_recovers_ols_without_outliers(self):
        X, y = generate_linear_data(
            n_samples=200, n_features=3, noise=0.01, random_seed=1
        )
        huber = HuberRegressionModel(
            input_dim=3, delta=5.0, learning_rate=0.05, epochs=500
        )
        huber.train(X, y)
        assert np.mean((huber.predict(X) - y) ** 2) < 0.05

    def test_parameter_roundtrip(self):
        X, y = generate_linear_data(n_samples=100, n_features=3, random_seed=2)
        model = HuberRegressionModel(input_dim=3)
        model.train(X, y)
        clone = HuberRegressionModel(input_dim=3)
        clone.set_parameters(model.get_parameters())
        np.testing.assert_array_almost_equal(clone.predict(X), model.predict(X))

    def test_invalid_delta_raises(self):
        with pytest.raises(ValueError, match="delta"):
            HuberRegressionModel(input_dim=3, delta=0.0)

    def test_empty_train_no_crash(self):
        HuberRegressionModel(input_dim=3).train(np.zeros((0, 3)), np.zeros(0))


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_data(n=100, d=3, seed=0):
    return generate_linear_data(n_samples=n, n_features=d, noise=0.05, random_seed=seed)


# ---------------------------------------------------------------------------
# LinearRegressionModel
# ---------------------------------------------------------------------------


class TestLinearRegressionModel:
    def test_predict_shape(self):
        model = LinearRegressionModel(input_dim=4)
        X = np.random.randn(10, 4)
        assert model.predict(X).shape == (10,)

    def test_train_reduces_loss(self):
        X, y = _make_data(200, 3)
        model = LinearRegressionModel(input_dim=3, learning_rate=0.05, epochs=100)
        loss_before = model.evaluate(X, y)
        model.train(X, y)
        loss_after = model.evaluate(X, y)
        assert loss_after < loss_before

    def test_get_set_parameters_roundtrip(self):
        model = LinearRegressionModel(input_dim=5)
        X, y = _make_data(50, 5)
        model.train(X, y)
        params = model.get_parameters().copy()
        # reset and restore
        model.set_parameters(np.zeros_like(params))
        assert not np.allclose(model.get_parameters(), params)
        model.set_parameters(params)
        np.testing.assert_array_equal(model.get_parameters(), params)

    def test_parameter_length(self):
        model = LinearRegressionModel(input_dim=7)
        assert len(model.get_parameters()) == 8  # 7 weights + 1 bias

    def test_train_empty_data_no_crash(self):
        """Regression: training on zero samples must not raise."""
        model = LinearRegressionModel(input_dim=3)
        model.train(np.zeros((0, 3)), np.zeros(0))  # should not raise

    def test_evaluate_empty_raises(self):
        model = LinearRegressionModel(input_dim=3)
        with pytest.raises(ValueError, match="empty"):
            model.evaluate(np.zeros((0, 3)), np.zeros(0))


# ---------------------------------------------------------------------------
# ClosedFormLinearRegressionModel
# ---------------------------------------------------------------------------


class TestClosedFormLinearRegressionModel:
    def test_near_perfect_fit_on_clean_data(self):
        """Closed-form solution on near-noise-free data should yield very low MSE."""
        X, y = _make_data(200, 4, seed=1)
        model = ClosedFormLinearRegressionModel(input_dim=4)
        model.train(X, y)
        assert model.evaluate(X, y) < 0.01

    def test_predict_shape(self):
        model = ClosedFormLinearRegressionModel(input_dim=3)
        X, y = _make_data(50, 3)
        model.train(X, y)
        assert model.predict(X).shape == (50,)

    def test_parameter_length(self):
        model = ClosedFormLinearRegressionModel(input_dim=6)
        # weights + bias appended as the last column
        assert len(model.get_parameters()) == 7

    def test_get_set_parameters_roundtrip(self):
        model = ClosedFormLinearRegressionModel(input_dim=4)
        X, y = _make_data(60, 4)
        model.train(X, y)
        params = model.get_parameters().copy()
        model.set_parameters(np.zeros_like(params))
        model.set_parameters(params)
        np.testing.assert_array_equal(model.get_parameters(), params)

    def test_train_empty_data_no_crash(self):
        model = ClosedFormLinearRegressionModel(input_dim=3)
        model.train(np.zeros((0, 3)), np.zeros(0))

    def test_evaluate_empty_raises(self):
        model = ClosedFormLinearRegressionModel(input_dim=3)
        with pytest.raises(ValueError, match="empty"):
            model.evaluate(np.zeros((0, 3)), np.zeros(0))

    def test_federated_avg_matches_centralized(self):
        """FedAvg on perfectly homogeneous data should match centralized solution."""
        X, y = _make_data(120, 3, seed=5)

        # Centralized model
        central = ClosedFormLinearRegressionModel(input_dim=3)
        central.train(X, y)

        # Each party trains on the full dataset (homogeneous case)
        n_parties = 3
        party_models = []
        for _ in range(n_parties):
            m = ClosedFormLinearRegressionModel(input_dim=3)
            m.train(X, y)
            party_models.append(m)

        avg_params = np.mean([m.get_parameters() for m in party_models], axis=0)
        fed_model = ClosedFormLinearRegressionModel(input_dim=3)
        fed_model.set_parameters(avg_params)

        # Both should produce nearly the same predictions
        np.testing.assert_allclose(central.predict(X), fed_model.predict(X), rtol=1e-5)
