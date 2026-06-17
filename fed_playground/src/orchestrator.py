"""Central orchestrator for the federated learning protocol.

The :class:`Orchestrator` coordinates all registered parties: it holds the
current global model, broadcasts it to parties before each round, and
aggregates their updates afterwards.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from .aggregation import AggregationStrategy
from .encryption import EncryptionScheme

if TYPE_CHECKING:
    from .attacks import Attack
    from .party import Party

logger = logging.getLogger(__name__)


class Orchestrator:
    """Central coordinator for a federated learning experiment.

    The orchestrator owns the global model state and drives two key operations
    per round:

    1. :meth:`distribute_model` — broadcast the current global parameters to
       all registered parties.
    2. :meth:`aggregate_models` — collect encrypted updates from all parties
       and combine them into a new global model.

    Args:
        aggregation_strategy: Strategy used to combine party updates.
        encryption_scheme: Scheme used during aggregation; must match the
            scheme used by all registered parties.
        initial_model_params: Starting global model parameters.  When
            ``None`` the first aggregation round initialises the global model.
    """

    def __init__(
        self,
        aggregation_strategy: AggregationStrategy,
        encryption_scheme: EncryptionScheme,
        initial_model_params: np.ndarray | None = None,
        attack: Attack | None = None,
        byzantine_ids: list[int] | None = None,
    ) -> None:
        self.aggregation_strategy = aggregation_strategy
        self.encryption_scheme = encryption_scheme
        self.global_model_params: np.ndarray | None = initial_model_params
        self.parties: list[Party] = []
        self.attack = attack
        self.byzantine_ids = byzantine_ids or []

    def register_party(self, party: Party) -> None:
        """Add a party to the list of participants.

        Args:
            party: The :class:`~fed_playground.src.party.Party` to register.
        """
        self.parties.append(party)

    def distribute_model(self) -> None:
        """Send the current global model to all registered parties.

        If no global model has been set yet (``global_model_params is None``),
        this is a no-op — parties retain whatever parameters they were
        initialised with.
        """
        if self.global_model_params is None:
            logger.debug("distribute_model called with no global model; skipping.")
            return

        for party in self.parties:
            party.update_model(self.global_model_params)

    def aggregate_models(self) -> None:
        """Collect party updates and compute the new global model.

        Each party's :meth:`~fed_playground.src.party.Party.get_encrypted_model`
        is called, the results are passed to the aggregation strategy, and
        ``global_model_params`` is updated with the result.

        Raises:
            RuntimeError: If no parties are registered.
        """
        if not self.parties:
            raise RuntimeError("No parties registered; cannot aggregate.")

        encrypted_models = [party.get_encrypted_model() for party in self.parties]
        if self.attack is not None and self.byzantine_ids:
            encrypted_models = self.attack.corrupt(encrypted_models, self.byzantine_ids)
        self.global_model_params = self.aggregation_strategy.aggregate(
            encrypted_models, self.encryption_scheme
        )
