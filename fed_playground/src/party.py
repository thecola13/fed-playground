"""Federated learning party (client) implementation.

A :class:`Party` holds a private local dataset, owns a local model, and
participates in each federated round by training locally and returning
(possibly encrypted) parameter updates to the orchestrator.
"""

import logging
from typing import Any

import numpy as np

from .encryption import EncryptionScheme
from .models import Model

logger = logging.getLogger(__name__)


class Party:
    """A single participant in the federated learning protocol.

    Each party holds private data that is never shared directly.  In every
    round it:

    1. Receives the global model from the :class:`~fed_playground.src.orchestrator.Orchestrator`.
    2. Fine-tunes that model on its local data.
    3. Returns (optionally encrypted) parameter updates.

    Args:
        party_id: Unique integer identifier for this party.
        model: Initialised :class:`~fed_playground.src.models.Model` instance.
        data: Tuple ``(X_train, y_train)`` of local training data.
        encryption_scheme: Scheme used to encrypt outgoing parameters.
    """

    def __init__(
        self,
        party_id: int,
        model: Model,
        data: tuple[np.ndarray, np.ndarray],
        encryption_scheme: EncryptionScheme,
    ) -> None:
        self.party_id = party_id
        self.model = model
        self.X_train, self.y_train = data
        self.encryption_scheme = encryption_scheme

    def train_local_model(self) -> None:
        """Train the local model on this party's private data."""
        self.model.train(self.X_train, self.y_train)

    def get_encrypted_model(self) -> Any:
        """Return the current model parameters, encrypted by the active scheme.

        Returns:
            Encrypted (or plaintext for :class:`~fed_playground.src.encryption.NoEncryption`)
            parameter vector.
        """
        params = self.model.get_parameters()
        return self.encryption_scheme.encrypt(params)

    def update_model(self, global_model_params: Any) -> None:
        """Update the local model with new global parameters.

        If the incoming parameters are encrypted (e.g. the orchestrator
        forwarded an encrypted global model), they are decrypted first using
        this party's encryption scheme.

        Args:
            global_model_params: Parameters as received from the orchestrator —
                may be a plaintext numpy array or an encrypted object depending
                on the protocol.

        Raises:
            TypeError: If decryption succeeds but the result cannot be passed
                to :meth:`~fed_playground.src.models.Model.set_parameters`.
        """
        try:
            params = self.encryption_scheme.decrypt(global_model_params)
        except (TypeError, ValueError, AttributeError) as exc:
            # The scheme reported that these parameters are already plaintext
            # or decryption is not applicable.  Trust the incoming value.
            logger.debug(
                "Party %d: decrypt raised %s — treating params as plaintext.",
                self.party_id,
                type(exc).__name__,
            )
            params = global_model_params

        self.model.set_parameters(params)

    def evaluate(
        self,
        X: np.ndarray | None = None,
        y: np.ndarray | None = None,
    ) -> float:
        """Evaluate the local model and return MSE.

        Args:
            X: Feature matrix for evaluation.  When ``None`` the local
                training data is used.
            y: Target vector for evaluation.  When ``None`` the local
                training target is used.

        Returns:
            Mean Squared Error on the provided (or local training) data.
        """
        if X is None or y is None:
            X, y = self.X_train, self.y_train
        return self.model.evaluate(X, y)
