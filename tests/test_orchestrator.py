"""Tests for fed_playground.src.orchestrator."""

import numpy as np
import pytest

from fed_playground.src.aggregation import MeanAggregation
from fed_playground.src.encryption import NoEncryption
from fed_playground.src.models import ClosedFormLinearRegressionModel
from fed_playground.src.orchestrator import Orchestrator
from fed_playground.src.party import Party
from fed_playground.src.utils_data import generate_linear_data


def _make_orchestrator(initial_params=None):
    return Orchestrator(
        aggregation_strategy=MeanAggregation(),
        encryption_scheme=NoEncryption(),
        initial_model_params=initial_params,
    )


def _make_party(party_id=0, seed=0):
    X, y = generate_linear_data(40, 3, random_seed=seed)
    model = ClosedFormLinearRegressionModel(input_dim=3)
    return Party(party_id, model, (X, y), NoEncryption())


class TestOrchestrator:
    def test_distribute_model_updates_parties(self):
        params = np.array([1.0, 2.0, 3.0, 0.5])
        orch = _make_orchestrator(initial_params=params)
        party = _make_party()
        orch.register_party(party)

        orch.distribute_model()
        np.testing.assert_array_equal(party.model.get_parameters(), params)

    def test_distribute_with_no_model_is_noop(self):
        orch = _make_orchestrator(initial_params=None)
        party = _make_party()
        orch.register_party(party)
        # Should not raise and should not modify party parameters
        original = party.model.get_parameters().copy()
        orch.distribute_model()
        np.testing.assert_array_equal(party.model.get_parameters(), original)

    def test_aggregate_updates_global_params(self):
        orch = _make_orchestrator(initial_params=np.zeros(4))
        parties = [_make_party(i, seed=i) for i in range(3)]
        for p in parties:
            p.train_local_model()
            orch.register_party(p)

        orch.aggregate_models()
        assert orch.global_model_params is not None
        assert orch.global_model_params.shape == (4,)

    def test_aggregate_no_parties_raises(self):
        orch = _make_orchestrator()
        with pytest.raises(RuntimeError, match="No parties"):
            orch.aggregate_models()

    def test_aggregate_equals_mean(self):
        """Aggregated params should equal the mean of party params."""
        params_list = [
            np.array([float(i)] * 4) for i in range(1, 4)
        ]  # [1,1,1,1], [2,2,2,2], [3,3,3,3]
        expected_mean = np.array([2.0, 2.0, 2.0, 2.0])

        orch = _make_orchestrator(initial_params=np.zeros(4))
        for i, params in enumerate(params_list):
            X, y = generate_linear_data(20, 3, random_seed=i)
            model = ClosedFormLinearRegressionModel(input_dim=3)
            model.set_parameters(params)
            party = Party(i, model, (X, y), NoEncryption())
            orch.register_party(party)

        orch.aggregate_models()
        np.testing.assert_array_almost_equal(orch.global_model_params, expected_mean)
