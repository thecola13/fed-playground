"""Tests for fed_playground.src.party."""

import numpy as np
import pytest

from fed_playground.src.encryption import NoEncryption
from fed_playground.src.models import ClosedFormLinearRegressionModel
from fed_playground.src.party import Party
from fed_playground.src.utils_data import generate_linear_data


def _make_party(seed=0):
    X, y = generate_linear_data(60, 3, random_seed=seed)
    model = ClosedFormLinearRegressionModel(input_dim=3)
    return Party(
        party_id=0,
        model=model,
        data=(X, y),
        encryption_scheme=NoEncryption(),
    )


class TestParty:
    def test_train_local_model(self):
        party = _make_party()
        loss_before = party.evaluate()
        party.train_local_model()
        loss_after = party.evaluate()
        assert loss_after <= loss_before

    def test_get_encrypted_model_shape(self):
        party = _make_party()
        party.train_local_model()
        enc = party.get_encrypted_model()
        assert isinstance(enc, np.ndarray)
        assert enc.shape == (4,)  # 3 weights + 1 bias

    def test_update_model(self):
        party = _make_party()
        new_params = np.array([1.0, 2.0, 3.0, 4.0])
        party.update_model(new_params)
        np.testing.assert_array_equal(party.model.get_parameters(), new_params)

    def test_evaluate_with_custom_data(self):
        party = _make_party()
        party.train_local_model()
        X_test, y_test = generate_linear_data(20, 3, random_seed=99)
        loss = party.evaluate(X_test, y_test)
        assert isinstance(loss, float)
        assert loss >= 0.0

    def test_evaluate_defaults_to_local_data(self):
        party = _make_party()
        party.train_local_model()
        loss_default = party.evaluate()
        X, y = party.X_train, party.y_train
        loss_explicit = party.evaluate(X, y)
        assert loss_default == pytest.approx(loss_explicit)

    def test_update_model_graceful_on_bad_decrypt(self):
        """Regression: update_model must not raise if decrypt raises TypeError."""
        from fed_playground.src.encryption import EncryptionScheme

        class BadDecryptScheme(EncryptionScheme):
            def encrypt(self, params):
                return params

            def decrypt(self, enc):
                raise TypeError("simulated failure")

            def aggregate(self, lst):
                return lst[0]

        X, y = generate_linear_data(40, 3, random_seed=1)
        model = ClosedFormLinearRegressionModel(input_dim=3)
        party = Party(0, model, (X, y), BadDecryptScheme())
        params = np.array([1.0, 2.0, 3.0, 0.5])
        party.update_model(params)  # must not raise
        np.testing.assert_array_equal(party.model.get_parameters(), params)
