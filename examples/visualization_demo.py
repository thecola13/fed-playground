"""Federated learning visualization demo.

Demonstrates all three built-in visualizers:

* :class:`~fed_playground.TrainingHistoryVisualizer` — loss curves over rounds.
* :class:`~fed_playground.ComparisonVisualizer` — bar chart comparing model
  types (centralized vs. federated vs. local).
* :class:`~fed_playground.DivergenceVisualizer` — divergence metrics across
  different party counts.

By default plots are displayed interactively.  Pass ``--save-dir <path>`` to
write PNG files instead.

Usage::

    python examples/visualization_demo.py
    python examples/visualization_demo.py --save-dir ./plots
"""

import argparse
import logging
import os
import sys
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from fed_playground import (
    ClosedFormLinearRegressionModel,
    ComparisonVisualizer,
    DataLoader,
    DivergenceVisualizer,
    Environment,
    MeanAggregation,
    NoEncryption,
    TrainingHistoryVisualizer,
)
from fed_playground.src.utils_data import generate_linear_data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)


def demo_training_history(save_dir: Optional[str], data_file: Optional[str]) -> None:
    """Run a short simulation and plot the per-round loss curves."""
    print("\n=== Training History Visualizer ===")

    if data_file and os.path.exists(data_file):
        loader: Optional[DataLoader] = DataLoader(
            file_path=data_file, target_column="target"
        )
    else:
        loader = None

    env = Environment(
        n_parties=3,
        encryption_scheme=NoEncryption(),
        aggregation_strategy=MeanAggregation(),
        model_class=ClosedFormLinearRegressionModel,
        n_features=5,
        n_samples=200,
        data_loader=loader,
    )
    history = env.run_simulation(rounds=10)

    viz = TrainingHistoryVisualizer(save_dir=save_dir)
    viz.plot(
        data={
            "Global Loss": history["global_loss"],
            "Avg Party Loss": history["party_loss"],
        },
        title="Federated Learning — Training History",
        xlabel="Round",
        ylabel="Loss (MSE)",
        filename="training_history.png",
    )

    final_gl = history["global_loss"][-1]
    final_pl = history["party_loss"][-1]
    print(f"  Final global test MSE : {final_gl:.6f}")
    print(f"  Final avg party MSE   : {final_pl:.6f}")


def demo_comparison(save_dir: Optional[str]) -> None:
    """Compare centralized, federated, and local-only model performance."""
    print("\n=== Comparison Visualizer ===")

    X, y = generate_linear_data(n_samples=1000, n_features=5, noise=0.1)
    X_train, y_train = X[:800], y[:800]
    X_test, y_test = X[800:], y[800:]

    centralized = ClosedFormLinearRegressionModel(input_dim=5)
    centralized.train(X_train, y_train)
    centralized_mse = centralized.evaluate(X_test, y_test)

    env = Environment(
        n_parties=5,
        encryption_scheme=NoEncryption(),
        aggregation_strategy=MeanAggregation(),
        model_class=ClosedFormLinearRegressionModel,
        data_loader=DataLoader(X=X_train, y=y_train),
    )
    env.run_simulation(rounds=5, test_data=(X_test, y_test))

    federated = ClosedFormLinearRegressionModel(input_dim=5)
    federated.set_parameters(env.orchestrator.global_model_params)
    federated_mse = federated.evaluate(X_test, y_test)

    local_mse = env.parties[0].model.evaluate(X_test, y_test)

    results = {
        "Centralized": centralized_mse,
        "Federated (5 parties)": federated_mse,
        "Local (party 0)": local_mse,
    }

    print(f"  Centralized MSE           : {centralized_mse:.6f}")
    print(f"  Federated MSE (5 parties) : {federated_mse:.6f}")
    print(f"  Local MSE (party 0)       : {local_mse:.6f}")

    viz = ComparisonVisualizer(save_dir=save_dir)
    viz.plot(
        data=results,
        title="Model Performance Comparison",
        xlabel="Model Type",
        ylabel="Test MSE",
        filename="model_comparison.png",
        color=["green", "steelblue", "orange"],
    )


def demo_divergence(save_dir: Optional[str]) -> None:
    """Analyse how divergence from centralized training scales with party count."""
    print("\n=== Divergence Visualizer ===")

    X, y = generate_linear_data(n_samples=2000, n_features=5, noise=0.1)
    X_train, y_train = X[:1600], y[:1600]
    X_test, y_test = X[1600:], y[1600:]

    viz = DivergenceVisualizer(save_dir=save_dir)

    for n_parties in [2, 4, 6, 8]:
        print(f"  Running {n_parties} parties …", end="", flush=True)

        centralized = ClosedFormLinearRegressionModel(input_dim=5)
        centralized.train(X_train, y_train)
        gen_mse = centralized.evaluate(X_test, y_test)

        metrics_per_round = []
        for _ in range(5):
            env = Environment(
                n_parties=n_parties,
                encryption_scheme=NoEncryption(),
                aggregation_strategy=MeanAggregation(),
                model_class=ClosedFormLinearRegressionModel,
                data_loader=DataLoader(X=X_train, y=y_train),
            )
            env.setup()
            env.run_round()

            fed = ClosedFormLinearRegressionModel(input_dim=5)
            fed.set_parameters(env.orchestrator.global_model_params)
            fed_mse = fed.evaluate(X_test, y_test)

            w_fed = fed.get_parameters()
            w_gen = centralized.get_parameters()
            norm_diff = float(np.linalg.norm(w_fed - w_gen))
            mse_diff = fed_mse - gen_mse
            mse_ratio = fed_mse / gen_mse if gen_mse != 0 else float("inf")

            local_metrics: dict[str, list] = {
                "local_mse": [],
                "local_normdiff": [],
                "local_msediff": [],
                "local_mseratio": [],
            }
            for party in env.parties:
                lm = party.model
                l_mse = lm.evaluate(X_test, y_test)
                w_local = lm.get_parameters()
                local_metrics["local_mse"].append(l_mse)
                local_metrics["local_normdiff"].append(
                    float(np.linalg.norm(w_local - w_gen))
                )
                local_metrics["local_msediff"].append(l_mse - gen_mse)
                local_metrics["local_mseratio"].append(
                    l_mse / gen_mse if gen_mse != 0 else float("inf")
                )

            metrics_per_round.append(
                {
                    "mse": fed_mse,
                    "general_mse": gen_mse,
                    "normdiff": norm_diff,
                    "msediff": mse_diff,
                    "mseratio": mse_ratio,
                    **local_metrics,
                }
            )

        viz.add_result(n_parties, metrics_per_round)
        print(f" done (centralized MSE={gen_mse:.4f})")

    viz.plot(
        x_label="Number of Parties",
        title_suffix="Party Count",
        show_local_models=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualization system demo.")
    parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="Directory to save PNG plots.  Omit to display interactively.",
    )
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_file = os.path.join(script_dir, "test_data.csv")

    print("=" * 60)
    print("Federated Learning Visualization Demo")
    print("=" * 60)
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        print(f"Saving plots to: {args.save_dir}")
    else:
        print("Displaying plots interactively (pass --save-dir to save instead).")

    demo_training_history(args.save_dir, data_file)
    demo_comparison(args.save_dir)
    demo_divergence(args.save_dir)

    print("\n" + "=" * 60)
    print("All demos complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
