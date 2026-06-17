"""Tests for fed_playground.src.benchmark."""

import numpy as np

from fed_playground.src.aggregation import KrumAggregation, MeanAggregation
from fed_playground.src.attacks import NoAttack, SignFlipAttack
from fed_playground.src.benchmark import run_benchmark
from fed_playground.src.encryption import PairwiseMaskingEncryption
from fed_playground.src.models import ClosedFormLinearRegressionModel


def test_grid_produces_one_row_per_combo():
    df = run_benchmark(
        models=[ClosedFormLinearRegressionModel],
        aggregations=[MeanAggregation(), KrumAggregation(n_byzantine=1)],
        attacks=[NoAttack(), SignFlipAttack(scale=10)],
        n_byzantine=(1,),
        n_parties=6,
        rounds=4,
    )
    assert len(df) == 1 * 2 * 1 * 2 * 1  # models x agg x enc x attack x n_byz
    assert set(df.columns) >= {
        "model",
        "aggregation",
        "encryption",
        "attack",
        "n_byzantine",
        "final_loss",
        "status",
    }
    assert (df["status"] == "ok").all()


def test_robust_aggregator_beats_mean_under_attack():
    df = run_benchmark(
        models=[ClosedFormLinearRegressionModel],
        aggregations=[MeanAggregation(), KrumAggregation(n_byzantine=1)],
        attacks=[SignFlipAttack(scale=10)],
        n_byzantine=(1,),
        n_parties=6,
        rounds=5,
    )
    loss = df.set_index("aggregation")["final_loss"]
    assert loss["KrumAggregation"] < loss["MeanAggregation"]


def test_incompatible_cell_is_recorded_not_raised():
    # Pairwise masking (is_linear_only) x Krum (order/distance) must not crash.
    df = run_benchmark(
        models=[ClosedFormLinearRegressionModel],
        aggregations=[KrumAggregation(n_byzantine=1)],
        encryptions=[PairwiseMaskingEncryption(n_parties=5)],
        n_parties=5,
        rounds=2,
    )
    assert len(df) == 1
    assert np.isnan(df.loc[0, "final_loss"])
    assert df.loc[0, "status"].startswith("incompatible")
