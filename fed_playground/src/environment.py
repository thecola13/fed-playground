"""High-level federated learning simulation environment.

:class:`Environment` orchestrates the full lifecycle of a federated learning
experiment: data preparation, party initialisation, and round execution.  It
is the primary entry point for users of the library.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

from .aggregation import AggregationStrategy, MeanAggregation
from .dataloader import DataLoader
from .encryption import EncryptionScheme, NoEncryption
from .models import LinearRegressionModel, Model
from .orchestrator import Orchestrator
from .party import Party
from .utils_data import generate_linear_data, split_data

if TYPE_CHECKING:
    from .attacks import Attack

logger = logging.getLogger(__name__)

# Fraction of samples withheld for global model evaluation.
_TEST_SPLIT = 0.2


class Environment:
    """Federated learning simulation environment.

    Manages data loading, party creation, and round execution.  After
    :meth:`run_simulation` (or manually calling :meth:`setup` then
    :meth:`run_round`) the training history is available in
    ``self.history``.

    Args:
        n_parties: Number of federated parties (clients).
        encryption_scheme: Encryption scheme applied to model parameters
            before transmission.  Defaults to :class:`~fed_playground.src.encryption.NoEncryption`.
        aggregation_strategy: Strategy for combining party updates.  Defaults
            to :class:`~fed_playground.src.aggregation.MeanAggregation`.
        n_features: Number of input features.  Required when *data_loader* is
            ``None`` (synthetic data generation).
        n_samples: Number of samples to generate.  Required when *data_loader*
            is ``None``.
        model_class: :class:`~fed_playground.src.models.Model` subclass to
            instantiate for each party and the global model.
        model_params: Keyword arguments forwarded to *model_class*
            (besides ``input_dim``).
        data_loader: Optional :class:`~fed_playground.src.dataloader.DataLoader`
            that provides training data.  When ``None`` synthetic data is
            generated from *n_samples* and *n_features*.

    Raises:
        ValueError: If *n_parties* < 1.
        ValueError: If *data_loader* is ``None`` and either *n_features* or
            *n_samples* is 0.
    """

    def __init__(
        self,
        n_parties: int,
        encryption_scheme: EncryptionScheme | None = None,
        aggregation_strategy: AggregationStrategy | None = None,
        n_features: int = 0,
        n_samples: int = 0,
        model_class: type[Model] = LinearRegressionModel,
        model_params: dict[str, Any] | None = None,
        data_loader: DataLoader | None = None,
        attack: Attack | None = None,
        n_byzantine: int = 0,
        seed: int = 42,
    ) -> None:
        if n_parties < 1:
            raise ValueError(f"n_parties must be >= 1, got {n_parties}.")
        if not 0 <= n_byzantine < n_parties:
            raise ValueError(f"n_byzantine must be in [0, {n_parties}).")

        self.n_parties = n_parties
        self.n_features = n_features
        self.n_samples = n_samples
        self.encryption_scheme: EncryptionScheme = encryption_scheme or NoEncryption()
        self.aggregation_strategy: AggregationStrategy = (
            aggregation_strategy or MeanAggregation()
        )
        self.model_class = model_class
        self.model_params: dict[str, Any] = model_params or {}
        self.data_loader = data_loader
        self.seed = seed
        # First n_byzantine parties are adversarial; their updates are poisoned
        # by `attack` at aggregation time.
        self.attack = attack
        self.byzantine_ids = list(range(n_byzantine))

        self.parties: list[Party] = []
        self.orchestrator: Orchestrator | None = None
        self.test_data: tuple[np.ndarray, np.ndarray] | None = None

        self.history: dict[str, list] = {
            "global_loss": [],
            "party_loss": [],
        }

    def setup(self) -> None:
        """Initialise data, parties, and the orchestrator.

        Must be called before :meth:`run_round`.  :meth:`run_simulation`
        calls this automatically.

        Raises:
            ValueError: If synthetic data is requested but *n_features* or
                *n_samples* is 0.
        """
        if self.data_loader:
            X, y = self.data_loader.load()
            self.n_samples = X.shape[0]
            self.n_features = X.shape[1]
        else:
            if self.n_features == 0 or self.n_samples == 0:
                raise ValueError(
                    "Provide a data_loader or set both n_features and n_samples > 0."
                )
            X, y = generate_linear_data(
                self.n_samples, self.n_features, random_seed=self.seed
            )

        split_idx = int((1 - _TEST_SPLIT) * self.n_samples)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        self.test_data = (X_test, y_test)

        party_splits = split_data(X_train, y_train, self.n_parties)

        self.parties = []
        for i, data_split in enumerate(party_splits):
            model = self.model_class(input_dim=self.n_features, **self.model_params)
            self.parties.append(
                Party(
                    party_id=i,
                    model=model,
                    data=data_split,
                    encryption_scheme=self.encryption_scheme,
                )
            )

        initial_model = self.model_class(input_dim=self.n_features, **self.model_params)
        self.orchestrator = Orchestrator(
            aggregation_strategy=self.aggregation_strategy,
            encryption_scheme=self.encryption_scheme,
            initial_model_params=initial_model.get_parameters(),
            attack=self.attack,
            byzantine_ids=self.byzantine_ids,
        )
        for party in self.parties:
            self.orchestrator.register_party(party)

    def run_round(self) -> None:
        """Execute one complete federated learning round.

        Round steps:

        1. Orchestrator broadcasts the current global model to all parties.
        2. Each party trains locally and its loss is recorded.
        3. Orchestrator aggregates party updates into a new global model.
        4. The global model is evaluated on the held-out test set.

        The recorded metrics are appended to ``self.history``.

        Raises:
            RuntimeError: If :meth:`setup` has not been called.
        """
        if self.orchestrator is None:
            raise RuntimeError("Call setup() before run_round().")

        self.orchestrator.distribute_model()

        avg_party_loss = 0.0
        for party in self.parties:
            party.train_local_model()
            avg_party_loss += party.evaluate()
        avg_party_loss /= self.n_parties
        self.history["party_loss"].append(avg_party_loss)

        self.orchestrator.aggregate_models()

        global_params = self.orchestrator.global_model_params
        if not isinstance(global_params, np.ndarray):
            try:
                global_params = self.encryption_scheme.decrypt(global_params)
            except (TypeError, ValueError) as exc:
                logger.warning(
                    "Cannot decrypt global model for evaluation: %s — "
                    "recording None for this round.",
                    exc,
                )
                global_params = None

        if isinstance(global_params, np.ndarray):
            eval_model = self.model_class(
                input_dim=self.n_features, **self.model_params
            )
            eval_model.set_parameters(global_params)
            X_test, y_test = self.test_data  # type: ignore[misc]
            global_loss: float | None = eval_model.evaluate(X_test, y_test)
        else:
            global_loss = None

        self.history["global_loss"].append(global_loss)

    def run_simulation(
        self,
        rounds: int = 10,
        test_data: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> dict[str, list]:
        """Run a complete federated learning simulation.

        Calls :meth:`setup` then executes *rounds* federated rounds, logging
        progress after each one.

        Args:
            rounds: Number of federated rounds to execute.
            test_data: Optional ``(X_test, y_test)`` tuple to override the
                held-out test set created during :meth:`setup`.

        Returns:
            History dictionary with keys ``"global_loss"`` and
            ``"party_loss"``, each mapping to a list of per-round values.
        """
        self.setup()

        if test_data is not None:
            self.test_data = test_data

        for r in range(rounds):
            self.run_round()
            global_loss = self.history["global_loss"][-1]
            global_loss_str = (
                f"{global_loss:.4f}" if global_loss is not None else "N/A (encrypted)"
            )
            logger.info(
                "Round %d/%d — avg party loss: %.4f, global test loss: %s",
                r + 1,
                rounds,
                self.history["party_loss"][-1],
                global_loss_str,
            )

        return self.history
