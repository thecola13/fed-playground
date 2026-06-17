"""Grid benchmark runner over the swappable federated-learning components.

One row per (model x aggregation x encryption x attack x n_byzantine) combo.
Reuses :class:`~fed_playground.src.environment.Environment` for each run, so the
benchmark adds no new simulation logic — just the sweep and a tidy table.
"""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING, Any

import pandas as pd

from .aggregation import AggregationStrategy, MeanAggregation
from .attacks import Attack, NoAttack
from .encryption import EncryptionScheme, NoEncryption
from .environment import Environment
from .models import LinearRegressionModel, Model

if TYPE_CHECKING:
    from .dataloader import DataLoader


def run_benchmark(
    *,
    models: list[type[Model]] | None = None,
    aggregations: list[AggregationStrategy] | None = None,
    encryptions: list[EncryptionScheme] | None = None,
    attacks: list[Attack] | None = None,
    n_byzantine: tuple[int, ...] = (0,),
    n_parties: int = 5,
    rounds: int = 10,
    n_features: int = 4,
    n_samples: int = 500,
    model_params: dict[str, Any] | None = None,
    data_loader: DataLoader | None = None,
) -> pd.DataFrame:
    """Run every component combination and return a tidy results DataFrame.

    Each axis defaults to a single sensible value, so you only pass the axes you
    want to sweep.  Combinations that are *incompatible* — e.g. a masking
    encryption (``is_linear_only``) paired with a distance/order aggregator —
    are caught and recorded with ``final_loss = NaN`` and ``status`` describing
    the error, rather than crashing the sweep.  Those NaN cells *are* the
    privacy x robustness frontier.

    Args:
        models: ``Model`` subclasses to sweep (default ``[LinearRegressionModel]``).
        aggregations: aggregation strategy instances (default ``[MeanAggregation()]``).
        encryptions: encryption scheme instances (default ``[NoEncryption()]``).
        attacks: attack instances (default ``[NoAttack()]``).
        n_byzantine: numbers of Byzantine parties to sweep (default ``(0,)``).
        n_parties, rounds, n_features, n_samples: fixed simulation settings.
        model_params: kwargs forwarded to every model.
        data_loader: optional shared dataset; when ``None`` each run uses the same
            seeded synthetic data, so rows are comparable.

    Returns:
        DataFrame with columns ``model, aggregation, encryption, attack,
        n_byzantine, final_loss, status``.
    """
    models = models or [LinearRegressionModel]
    aggregations = aggregations or [MeanAggregation()]
    encryptions = encryptions or [NoEncryption()]
    attacks = attacks or [NoAttack()]

    rows: list[dict[str, Any]] = []
    for model, agg, enc, atk, nb in itertools.product(
        models, aggregations, encryptions, attacks, n_byzantine
    ):
        row: dict[str, Any] = {
            "model": model.__name__,
            "aggregation": type(agg).__name__,
            "encryption": type(enc).__name__,
            "attack": type(atk).__name__,
            "n_byzantine": nb,
        }
        try:
            env = Environment(
                n_parties=n_parties,
                n_features=n_features,
                n_samples=n_samples,
                encryption_scheme=enc,
                aggregation_strategy=agg,
                model_class=model,
                model_params=model_params,
                data_loader=data_loader,
                attack=atk,
                n_byzantine=nb,
            )
            history = env.run_simulation(rounds=rounds)
            row["final_loss"] = history["global_loss"][-1]
            row["status"] = "ok"
        except ValueError as exc:  # e.g. masking scheme x order-statistic aggregator
            row["final_loss"] = float("nan")
            row["status"] = f"incompatible: {exc}"[:80]
        rows.append(row)

    return pd.DataFrame(rows)
