"""Tests for fed_playground.src.environment."""

import pytest

from fed_playground.src.aggregation import MeanAggregation
from fed_playground.src.dataloader import DataLoader
from fed_playground.src.encryption import NoEncryption
from fed_playground.src.environment import Environment
from fed_playground.src.models import (
    ClosedFormLinearRegressionModel,
    LinearRegressionModel,
)
from fed_playground.src.utils_data import generate_linear_data


def _make_env(**kwargs):
    defaults = dict(
        n_parties=3,
        encryption_scheme=NoEncryption(),
        aggregation_strategy=MeanAggregation(),
        n_features=4,
        n_samples=120,
        model_class=ClosedFormLinearRegressionModel,
    )
    defaults.update(kwargs)
    return Environment(**defaults)


class TestEnvironmentSetup:
    def test_setup_creates_parties(self):
        env = _make_env()
        env.setup()
        assert len(env.parties) == 3

    def test_setup_creates_orchestrator(self):
        env = _make_env()
        env.setup()
        assert env.orchestrator is not None

    def test_setup_with_data_loader(self):
        X, y = generate_linear_data(80, 5, random_seed=1)
        loader = DataLoader(X=X, y=y)
        env = _make_env(n_features=0, n_samples=0, data_loader=loader)
        env.setup()
        assert env.n_features == 5
        assert env.n_samples == 80

    def test_invalid_n_parties_raises(self):
        with pytest.raises(ValueError, match="n_parties"):
            Environment(n_parties=0, n_features=4, n_samples=100)

    def test_setup_without_data_requires_dimensions(self):
        env = Environment(n_parties=2, n_features=0, n_samples=0)
        with pytest.raises(ValueError, match="data_loader"):
            env.setup()


class TestEnvironmentRun:
    def test_run_round_populates_history(self):
        env = _make_env()
        env.setup()
        env.run_round()
        assert len(env.history["global_loss"]) == 1
        assert len(env.history["party_loss"]) == 1

    def test_run_round_without_setup_raises(self):
        env = _make_env()
        with pytest.raises(RuntimeError, match="setup"):
            env.run_round()

    def test_run_simulation_returns_history(self):
        env = _make_env()
        history = env.run_simulation(rounds=3)
        assert len(history["global_loss"]) == 3
        assert len(history["party_loss"]) == 3

    def test_global_loss_decreases_over_rounds(self):
        """Federated learning should reduce global test MSE across rounds."""
        env = Environment(
            n_parties=3,
            encryption_scheme=NoEncryption(),
            aggregation_strategy=MeanAggregation(),
            n_features=4,
            n_samples=300,
            model_class=LinearRegressionModel,
            model_params={"learning_rate": 0.01, "epochs": 5},
        )
        history = env.run_simulation(rounds=10)
        losses = [loss for loss in history["global_loss"] if loss is not None]
        assert len(losses) > 0
        assert losses[-1] < losses[0]

    def test_custom_test_data_used(self):
        env = _make_env()
        X_test, y_test = generate_linear_data(30, 4, random_seed=99)
        history = env.run_simulation(rounds=2, test_data=(X_test, y_test))
        assert all(loss is not None for loss in history["global_loss"])

    def test_multiple_simulations_reset_history(self):
        env = _make_env()
        env.run_simulation(rounds=2)
        # Second call to run_simulation calls setup() which resets state
        env.history = {"global_loss": [], "party_loss": []}
        env.run_simulation(rounds=3)
        assert len(env.history["global_loss"]) == 3
