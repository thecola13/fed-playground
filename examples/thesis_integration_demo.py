"""
Master Thesis Integration Demo

Demonstrates compatibility between master thesis data/experiments
and the fed_playground library. Shows how to:
1. Load data in master thesis format
2. Run divergence experiments
3. Generate visualizations matching thesis results
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from typing import List
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fed_playground import (
    ClosedFormLinearRegressionModel,
    MeanAggregation,
    DataLoader,
    Environment,
    NoEncryption,
    DivergenceVisualizer
)

from thesis_utils import (
    get_thesis_data_path,
    parse_thesis_experiment_name,
    print_thesis_results_summary
)


def run_divergence_experiment(
    data_path: str,
    target: str,
    features: List[str],
    mode: str = "instances_diff",
    min_instances: int = 2,
    max_instances: int = 10,
    step_instances: int = 2,
    samples_per_instance: int = 8,
    min_data: int = 10,
    max_data: int = 100,
    step_data: int = 10,
    total_data_points: int = 120,
    rounds: int = 5,
    test_size: float = 0.2,
    random_state: int = 42,
    save_dir: str = "thesis_results",
    transpose: bool = False,
    verbosity: int = 1
):
    """
    Run divergence experiment matching master thesis methodology.

    Args:
        data_path: Path to data file
        target: Target variable name
        features: List of feature names
        mode: Experiment mode ("instances_diff", "data_diff", or "fixed_data")
        min_instances: Minimum number of parties (for instances_diff, fixed_data)
        max_instances: Maximum number of parties
        step_instances: Step size for parties
        samples_per_instance: Samples per party (for instances_diff)
        min_data: Minimum data per party (for data_diff)
        max_data: Maximum data per party (for data_diff)
        step_data: Step size for data (for data_diff)
        total_data_points: Total data points (for fixed_data)
        rounds: Number of FL rounds
        test_size: Test set proportion
        random_state: Random seed
        save_dir: Directory to save results
        transpose: Whether data is in transposed format
        verbosity: Verbosity level
    """
    if verbosity >= 1:
        print(f"\n{'='*60}")
        print(f"Master Thesis Integration Demo")
        print(f"{'='*60}")
        print(f"Data: {os.path.basename(data_path)}")
        print(f"Target: {target}")
        print(f"Features: {', '.join(features)}")
        print(f"Mode: {mode}")
        print(f"{'='*60}\n")

    # Load and split data
    loader = DataLoader(
        file_path=data_path,
        target_column=target,
        feature_columns=features,
        transpose=transpose
    )

    X, y = loader.load()

    # Train/test split
    np.random.seed(random_state)
    n = len(X)
    indices = np.random.permutation(n)
    split_idx = int(n * (1 - test_size))

    train_idx, test_idx = indices[:split_idx], indices[split_idx:]
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    if verbosity >= 1:
        print(f"Data split: {len(X_train)} train, {len(X_test)} test samples")

    # Train centralized model for comparison
    n_features = X_train.shape[1]
    general_model = ClosedFormLinearRegressionModel(input_dim=n_features)
    general_model.train(X_train, y_train)
    gen_mse = general_model.evaluate(X_test, y_test)

    if verbosity >= 1:
        print(f"Centralized model MSE: {gen_mse:.6f}\n")

    # Initialize visualizer
    visualizer = DivergenceVisualizer(save_dir=save_dir)

    # Run experiments based on mode
    if mode == "instances_diff":
        run_instances_diff(
            X_train, y_train, X_test, y_test,
            general_model, visualizer,
            features, target,
            min_instances, max_instances, step_instances,
            samples_per_instance, rounds, random_state,
            verbosity
        )
        visualizer.plot(x_label="Number of Parties", title_suffix="Party Count")

    elif mode == "data_diff":
        run_data_diff(
            X_train, y_train, X_test, y_test,
            general_model, visualizer,
            features, target,
            min_data, max_data, step_data,
            max_instances, rounds, random_state,
            verbosity
        )
        visualizer.plot(x_label="Data Per Party", title_suffix="Data Amount")

    elif mode == "fixed_data":
        run_fixed_data(
            X_train, y_train, X_test, y_test,
            general_model, visualizer,
            features, target,
            total_data_points, min_instances, max_instances,
            step_instances, rounds, random_state,
            verbosity
        )
        visualizer.plot(x_label="Number of Parties", title_suffix="Fixed Data Total")

    if verbosity >= 1:
        print(f"\n{'='*60}")
        print(f"Results saved to: {save_dir}")
        print(f"{'='*60}\n")


def run_instances_diff(
    X_train, y_train, X_test, y_test,
    general_model, visualizer,
    features, target,
    min_instances, max_instances, step_instances,
    samples_per_instance, rounds, random_state,
    verbosity
):
    """Run instances_diff experiment (vary number of parties)."""
    instance_range = range(min_instances, max_instances + 1, step_instances)

    for num_instances in tqdm(list(instance_range), desc="Varying instances"):
        total_samples = num_instances * samples_per_instance

        if len(X_train) < total_samples:
            if verbosity >= 1:
                print(f"Skipping {num_instances} instances (insufficient data)")
            continue

        metrics_per_round = run_fl_rounds(
            X_train[:total_samples], y_train[:total_samples],
            X_test, y_test,
            general_model, num_instances, rounds,
            features, target, random_state,
            verbosity >= 2
        )

        visualizer.add_result(num_instances, metrics_per_round)


def run_data_diff(
    X_train, y_train, X_test, y_test,
    general_model, visualizer,
    features, target,
    min_data, max_data, step_data,
    num_instances, rounds, random_state,
    verbosity
):
    """Run data_diff experiment (vary data per party)."""
    data_range = range(min_data, max_data + 1, step_data)

    for samples_per_instance in tqdm(list(data_range), desc="Varying data amount"):
        total_samples = num_instances * samples_per_instance

        if len(X_train) < total_samples:
            if verbosity >= 1:
                print(f"Skipping {samples_per_instance} samples (insufficient data)")
            continue

        metrics_per_round = run_fl_rounds(
            X_train[:total_samples], y_train[:total_samples],
            X_test, y_test,
            general_model, num_instances, rounds,
            features, target, random_state,
            verbosity >= 2
        )

        visualizer.add_result(samples_per_instance, metrics_per_round)


def run_fixed_data(
    X_train, y_train, X_test, y_test,
    general_model, visualizer,
    features, target,
    total_data_points, min_instances, max_instances,
    step_instances, rounds, random_state,
    verbosity
):
    """Run fixed_data experiment (fixed total data, vary parties)."""
    instance_range = range(min_instances, max_instances + 1, step_instances)

    for num_instances in tqdm(list(instance_range), desc="Varying instances (fixed data)"):
        samples_per_instance = total_data_points // num_instances

        if samples_per_instance < 1 or len(X_train) < total_data_points:
            if verbosity >= 1:
                print(f"Skipping {num_instances} instances")
            continue

        metrics_per_round = run_fl_rounds(
            X_train[:total_data_points], y_train[:total_data_points],
            X_test, y_test,
            general_model, num_instances, rounds,
            features, target, random_state,
            verbosity >= 2
        )

        visualizer.add_result(num_instances, metrics_per_round)


def run_fl_rounds(
    X_train, y_train, X_test, y_test,
    general_model, num_instances, rounds,
    features, target, random_state, verbose=False
):
    """Run multiple FL rounds and collect metrics."""
    metrics_per_round = []
    n_features = X_train.shape[1]
    gen_mse = general_model.evaluate(X_test, y_test)

    for round_idx in range(rounds):
        # Shuffle data for this round
        np.random.seed(random_state + round_idx)
        indices = np.random.permutation(len(X_train))
        X_shuffled = X_train[indices]
        y_shuffled = y_train[indices]

        # Create DataFrame for DataLoader
        df = pd.DataFrame(X_shuffled, columns=[f"feat_{i}" for i in range(n_features)])
        df[target] = y_shuffled

        # Run FL round
        loader = DataLoader(dataframe=df, target_column=target)

        env = Environment(
            n_parties=num_instances,
            encryption_scheme=NoEncryption(),
            aggregation_strategy=MeanAggregation(),
            model_class=ClosedFormLinearRegressionModel,
            data_loader=loader
        )

        env.setup()
        env.run_round()

        # Evaluate aggregated model
        agg_params = env.orchestrator.global_model_params
        avg_model = ClosedFormLinearRegressionModel(input_dim=n_features)
        avg_model.set_parameters(agg_params)
        fed_mse = avg_model.evaluate(X_test, y_test)

        # Calculate divergence metrics
        w_fed = avg_model.get_parameters()
        w_gen = general_model.get_parameters()

        round_metrics = {
            "mse": fed_mse,
            "general_mse": gen_mse,
            "normdiff": float(np.linalg.norm(w_fed - w_gen)),
            "msediff": float(fed_mse - gen_mse),
            "mseratio": float(fed_mse / gen_mse if gen_mse != 0 else np.inf),
        }

        # Collect local model metrics
        local_mse = []
        local_normdiff = []
        local_msediff = []
        local_mseratio = []

        for party in env.parties:
            l_model = party.model
            l_mse = l_model.evaluate(X_test, y_test)
            w_local = l_model.get_parameters()
            l_normdiff = np.linalg.norm(w_local - w_gen)

            local_mse.append(float(l_mse))
            local_normdiff.append(float(l_normdiff))
            local_msediff.append(float(l_mse - gen_mse))
            local_mseratio.append(float(l_mse / gen_mse if gen_mse != 0 else np.inf))

        round_metrics["local_mse"] = local_mse
        round_metrics["local_normdiff"] = local_normdiff
        round_metrics["local_msediff"] = local_msediff
        round_metrics["local_mseratio"] = local_mseratio

        metrics_per_round.append(round_metrics)

        if verbose:
            print(f"  Round {round_idx+1}: Fed MSE={fed_mse:.6f}, "
                  f"Diff={fed_mse - gen_mse:.6f}")

    return metrics_per_round


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Master Thesis Integration Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Synthetic data with instances_diff mode
  python thesis_integration_demo.py --data synthetic --target y --features x1 x2 x3 x4 --instances-diff

  # Metabric data with data_diff mode (using real gene names)
  python thesis_integration_demo.py --data metabric --target BRCA1 --features TP53 MKI67 FOXM1 KIF20A --data-diff

  # Custom data file
  python thesis_integration_demo.py --data-path /path/to/data.csv --target target_name --features feat1 feat2 --fixed-data

  # List all thesis results
  python thesis_integration_demo.py --list-results
        """
    )

    parser.add_argument("--data", type=str, choices=["synthetic", "metabric"],
                       help="Preset data source (synthetic or metabric)")
    parser.add_argument("--data-path", type=str,
                       help="Path to custom data file")
    parser.add_argument("--target", type=str, required=False,
                       help="Target variable name")
    parser.add_argument("--features", type=str, nargs="+", required=False,
                       help="Feature names")
    parser.add_argument("--transpose", action="store_true",
                       help="Transpose data (features as rows)")

    # Experiment modes
    parser.add_argument("--instances-diff", action="store_true",
                       help="Vary number of parties")
    parser.add_argument("--data-diff", action="store_true",
                       help="Vary data per party")
    parser.add_argument("--fixed-data", action="store_true",
                       help="Fixed total data, vary parties")

    # Parameters
    parser.add_argument("--min-instances", type=int, default=2)
    parser.add_argument("--max-instances", type=int, default=10)
    parser.add_argument("--step-instances", type=int, default=2)
    parser.add_argument("--samples-per-instance", type=int, default=8)
    parser.add_argument("--min-data", type=int, default=10)
    parser.add_argument("--max-data", type=int, default=100)
    parser.add_argument("--step-data", type=int, default=10)
    parser.add_argument("--total-data-points", type=int, default=120)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--save-dir", type=str, default="thesis_results")
    parser.add_argument("--verbosity", type=int, default=1)

    # Utility
    parser.add_argument("--list-results", action="store_true",
                       help="List all master thesis results")

    args = parser.parse_args()

    # Handle list-results
    if args.list_results:
        print_thesis_results_summary()
        return

    # Determine data path
    if args.data:
        data_path = get_thesis_data_path(args.data)
        transpose = True  # Thesis data is transposed
    elif args.data_path:
        data_path = args.data_path
        transpose = args.transpose
    else:
        parser.error("Must provide either --data or --data-path")

    # Validate required arguments
    if not args.target or not args.features:
        parser.error("--target and --features are required for experiments")

    # Determine mode
    mode_count = sum([args.instances_diff, args.data_diff, args.fixed_data])
    if mode_count == 0:
        print("No mode specified, defaulting to --instances-diff")
        mode = "instances_diff"
    elif mode_count > 1:
        parser.error("Specify only one mode: --instances-diff, --data-diff, or --fixed-data")
    else:
        if args.instances_diff:
            mode = "instances_diff"
        elif args.data_diff:
            mode = "data_diff"
        else:
            mode = "fixed_data"

    # Run experiment
    run_divergence_experiment(
        data_path=data_path,
        target=args.target,
        features=args.features,
        mode=mode,
        min_instances=args.min_instances,
        max_instances=args.max_instances,
        step_instances=args.step_instances,
        samples_per_instance=args.samples_per_instance,
        min_data=args.min_data,
        max_data=args.max_data,
        step_data=args.step_data,
        total_data_points=args.total_data_points,
        rounds=args.rounds,
        test_size=args.test_size,
        random_state=args.random_state,
        save_dir=args.save_dir,
        transpose=transpose,
        verbosity=args.verbosity
    )


if __name__ == "__main__":
    main()
