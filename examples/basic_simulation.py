"""Basic federated learning simulation.

Demonstrates how to set up an :class:`~fed_playground.Environment` with either
a CSV dataset or synthetic data, run a multi-round simulation, and inspect the
resulting training history.

Usage::

    # with the bundled test dataset
    python examples/basic_simulation.py

    # with synthetic data (no CSV required)
    python examples/basic_simulation.py --synthetic
"""

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fed_playground import (
    ClosedFormLinearRegressionModel,
    DataLoader,
    Environment,
    MeanAggregation,
    NoEncryption,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Basic FL simulation demo.")
    parser.add_argument("--parties", type=int, default=3, help="Number of FL parties.")
    parser.add_argument("--rounds", type=int, default=5, help="Number of FL rounds.")
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic data instead of the bundled CSV.",
    )
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_file = os.path.join(script_dir, "test_data.csv")

    if args.synthetic or not os.path.exists(data_file):
        if not args.synthetic:
            logging.warning("test_data.csv not found — falling back to synthetic data.")
        data_loader = None
        n_features, n_samples = 5, 200
        logging.info(
            "Using synthetic data: %d samples, %d features.", n_samples, n_features
        )
    else:
        data_loader = DataLoader(file_path=data_file, target_column="target")
        n_features, n_samples = 0, 0
        logging.info("Loading data from %s", data_file)

    env = Environment(
        n_parties=args.parties,
        n_features=n_features,
        n_samples=n_samples,
        encryption_scheme=NoEncryption(),
        aggregation_strategy=MeanAggregation(),
        model_class=ClosedFormLinearRegressionModel,
        data_loader=data_loader,
    )

    logging.info(
        "Starting simulation: %d parties, %d rounds.", args.parties, args.rounds
    )
    history = env.run_simulation(rounds=args.rounds)

    print("\n--- Results ---")
    print(f"{'Round':>5}  {'Party Loss':>12}  {'Global Loss':>12}")
    print("-" * 35)
    for i, (pl, gl) in enumerate(
        zip(history["party_loss"], history["global_loss"]), start=1
    ):
        gl_str = f"{gl:.6f}" if gl is not None else "N/A"
        print(f"{i:>5}  {pl:>12.6f}  {gl_str:>12}")

    final = history["global_loss"][-1]
    if final is not None:
        print(f"\nFinal global test MSE: {final:.6f}")


if __name__ == "__main__":
    main()
