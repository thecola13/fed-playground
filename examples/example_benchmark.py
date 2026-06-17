"""One-call benchmark sweep over the federated-learning component matrix.

Uses ``run_benchmark`` to evaluate every (aggregation x attack) combination and
prints a tidy results table — the Phase 1 benchmark engine. The same call scales
to sweeping models, encryption schemes, Byzantine counts, etc.

Run:
    uv run python examples/example_benchmark.py
"""

from __future__ import annotations

from fed_playground import (
    ALittleIsEnoughAttack,
    BulyanAggregation,
    CenteredClippingAggregation,
    ClosedFormLinearRegressionModel,
    GeometricMedianAggregation,
    IPMAttack,
    KrumAggregation,
    MeanAggregation,
    MedianAggregation,
    NoAttack,
    SignFlipAttack,
    run_benchmark,
)


def main() -> None:
    df = run_benchmark(
        models=[ClosedFormLinearRegressionModel],
        aggregations=[
            MeanAggregation(),
            MedianAggregation(),
            KrumAggregation(n_byzantine=2),
            BulyanAggregation(n_byzantine=2),
            GeometricMedianAggregation(),
            CenteredClippingAggregation(clip_radius=1.0, n_iters=5),
        ],
        attacks=[
            NoAttack(),
            SignFlipAttack(scale=10),
            IPMAttack(epsilon=2.0),
            ALittleIsEnoughAttack(z=3.0),
        ],
        n_byzantine=(2,),
        n_parties=11,
        rounds=8,
    )

    # Pivot to an attack x defense matrix of final global test MSE (lower better).
    matrix = df.pivot(index="aggregation", columns="attack", values="final_loss")
    print("\nFinal global test MSE — attack (columns) x defense (rows), lower is better:\n")
    print(matrix.round(3).to_string())
    print(
        "\nFedAvg (MeanAggregation) collapses under every attack; the robust\n"
        "aggregators stay near the clean optimum. IPM / ALittleIsEnough are the\n"
        "adaptive attacks designed to slip past distance/median defenses.\n"
    )


if __name__ == "__main__":
    main()
