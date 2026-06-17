"""Federated learning divergence analysis.

Quantifies how much a federated model diverges from a centralized (oracle)
model as a function of either the number of parties or the amount of data
per party.

Three analysis modes (mutually exclusive):

* ``--instances-diff``  — vary the number of parties; keep data per party fixed.
* ``--data-diff``       — vary data per party; keep number of parties fixed.
* ``--fixed-data``      — vary parties while holding *total* data constant.

Divergence metrics reported per configuration:

* **Norm difference**  — L2 distance between federated and centralized weights.
* **MSE difference**   — (federated MSE) − (centralized MSE).
* **MSE ratio**        — (federated MSE) / (centralized MSE).

Results are printed to stdout.  Pass ``--save-dir`` to additionally save plots.

Usage::

    python examples/divergence_analysis.py \\
        --data-path examples/test_data.csv \\
        --features feature_1 feature_2 feature_3 feature_4 feature_5 \\
        --target target \\
        --instances-diff --min-instances 2 --max-instances 8
"""

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import tqdm

from fed_playground import (
    ClosedFormLinearRegressionModel,
    DataLoader,
    DivergenceVisualizer,
    Environment,
    MeanAggregation,
    NoEncryption,
)

logging.basicConfig(
    level=logging.WARNING,  # tqdm handles progress; keep noise low
    format="%(levelname)s: %(message)s",
)


def make_supervised_table(
    df: pd.DataFrame,
    features: list[str],
    target: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """Validate and extract feature matrix and target from a DataFrame.

    Args:
        df: Source DataFrame.
        features: List of feature column names.
        target: Target column name.

    Returns:
        Tuple ``(X, y)`` as DataFrame and Series respectively.

    Raises:
        ValueError: If any feature or the target column is missing.
    """
    missing_feats = [f for f in features if f not in df.columns]
    if missing_feats:
        raise ValueError(f"Features not found in data: {missing_feats}")
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in data.")
    return df[features], df[target]


def train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Shuffle and split data into train and test sets.

    Args:
        X: Feature DataFrame.
        y: Target Series.
        test_size: Fraction of samples reserved for the test set.
        random_state: Seed for reproducibility.

    Returns:
        Tuple ``(X_train, X_test, y_train, y_test)``.
    """
    rng = np.random.default_rng(random_state)
    n = len(X)
    indices = rng.permutation(n)
    split_idx = int(n * (1 - test_size))
    train_idx, test_idx = indices[:split_idx], indices[split_idx:]
    return (
        X.iloc[train_idx],
        X.iloc[test_idx],
        y.iloc[train_idx],
        y.iloc[test_idx],
    )


def evaluate_model(
    model: ClosedFormLinearRegressionModel,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    general_model: ClosedFormLinearRegressionModel | None = None,
) -> dict:
    """Evaluate a model and optionally compute divergence from a reference.

    Args:
        model: Model to evaluate.
        X_test: Test feature matrix.
        y_test: Test target vector.
        general_model: Optional reference (centralized) model.  When provided,
            norm difference, MSE difference, and MSE ratio are also computed.

    Returns:
        Dictionary with key ``"mse"`` and (if *general_model* is provided)
        ``"normdiff"``, ``"msediff"``, and ``"mseratio"``.
    """
    X_np = X_test.to_numpy() if isinstance(X_test, pd.DataFrame) else X_test
    y_np = y_test.to_numpy() if isinstance(y_test, pd.Series) else y_test

    mse = model.evaluate(X_np, y_np)
    metrics: dict = {"mse": mse}

    if general_model is not None:
        w_local = model.get_parameters()
        w_gen = general_model.get_parameters()
        gen_mse = general_model.evaluate(X_np, y_np)
        metrics["normdiff"] = float(np.linalg.norm(w_local - w_gen))
        metrics["msediff"] = mse - gen_mse
        metrics["mseratio"] = mse / gen_mse if gen_mse != 0 else float("inf")

    return metrics


def do_one_experiment(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    num_instances: int,
    sample_per_instance: int,
    rounds: int,
    random_state: int,
    feature_columns: list[str],
    target_column: str,
    collect_local: bool = True,
) -> list[dict]:
    """Run *rounds* independent FL iterations for a given configuration.

    Each iteration re-shuffles the training data so that results reflect
    variance across different data orderings.

    Args:
        X_train: Training features.
        y_train: Training targets.
        X_test: Test features.
        y_test: Test targets.
        num_instances: Number of FL parties.
        sample_per_instance: Training samples allocated to each party.
        rounds: Number of independent iterations.
        random_state: Base seed; each round uses ``random_state + round_index``.
        feature_columns: Column names used as features.
        target_column: Column name used as the target.
        collect_local: When ``True``, per-party metrics are also collected.

    Returns:
        List of per-round metric dictionaries with keys:
        ``mse``, ``general_mse``, ``normdiff``, ``msediff``, ``mseratio``
        and optionally ``local_mse``, ``local_normdiff``, ``local_msediff``,
        ``local_mseratio``.

    Raises:
        ValueError: If there is not enough training data for the requested
            configuration.
    """
    end_total = num_instances * sample_per_instance
    if X_train.shape[0] < end_total:
        raise ValueError(
            f"Not enough training data: need {end_total}, have {X_train.shape[0]}."
        )

    n_features = X_train.shape[1]
    X_gen = X_train.iloc[:end_total].to_numpy()
    y_gen = y_train.iloc[:end_total].to_numpy()

    general_model = ClosedFormLinearRegressionModel(input_dim=n_features)
    general_model.train(X_gen, y_gen)
    general_mse = evaluate_model(general_model, X_test, y_test)["mse"]

    metrics_per_round: list[dict] = []
    train_df = pd.concat([X_train, y_train], axis=1)

    for r in tqdm.tqdm(
        range(rounds),
        leave=False,
        desc=f"Rounds (I={num_instances}, D={sample_per_instance})",
    ):
        shuffled = train_df.sample(frac=1, random_state=random_state + r)
        subset = shuffled.iloc[:end_total]

        loader = DataLoader(
            dataframe=subset,
            target_column=target_column,
            feature_columns=feature_columns,
        )
        env = Environment(
            n_parties=num_instances,
            encryption_scheme=NoEncryption(),
            aggregation_strategy=MeanAggregation(),
            model_class=ClosedFormLinearRegressionModel,
            data_loader=loader,
        )
        env.setup()
        env.run_round()

        avg_model = ClosedFormLinearRegressionModel(input_dim=n_features)
        avg_model.set_parameters(env.orchestrator.global_model_params)
        round_metrics = evaluate_model(avg_model, X_test, y_test, general_model)
        round_metrics["general_mse"] = general_mse

        if collect_local:
            local_mse, local_nd, local_md, local_mr = [], [], [], []
            for party in env.parties:
                m = evaluate_model(party.model, X_test, y_test, general_model)
                local_mse.append(m["mse"])
                local_nd.append(m["normdiff"])
                local_md.append(m["msediff"])
                local_mr.append(m["mseratio"])
            round_metrics["local_mse"] = local_mse
            round_metrics["local_normdiff"] = local_nd
            round_metrics["local_msediff"] = local_md
            round_metrics["local_mseratio"] = local_mr

        metrics_per_round.append(round_metrics)

    return metrics_per_round


def print_summary(plotter: DivergenceVisualizer, mode_label: str) -> None:
    """Print a summary table of divergence results to stdout.

    Args:
        plotter: Populated :class:`~fed_playground.DivergenceVisualizer`.
        mode_label: Human-readable label for the x-axis (e.g. ``"Parties"``).
    """
    print(f"\n{'─' * 70}")
    print(f"  Divergence Summary — {mode_label}")
    print(f"{'─' * 70}")
    header = f"  {'X':>8}  {'Fed MSE':>10}  {'Gen MSE':>10}  {'Norm Diff':>10}  {'MSE Ratio':>10}"
    print(header)
    print(f"  {'-'*8}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}")

    for x_val in sorted(plotter.results.keys()):
        rounds_list = plotter.results[x_val]
        avg_fed = float(np.mean([r["mse"] for r in rounds_list]))
        avg_gen = float(np.mean([r["general_mse"] for r in rounds_list]))
        avg_nd = float(np.mean([r["normdiff"] for r in rounds_list]))
        avg_mr = float(np.mean([r["mseratio"] for r in rounds_list]))
        print(
            f"  {x_val:>8}  {avg_fed:>10.4f}  {avg_gen:>10.4f}"
            f"  {avg_nd:>10.4f}  {avg_mr:>10.4f}"
        )
    print(f"{'─' * 70}\n")


def main(args: argparse.Namespace) -> None:
    if args.data_path:
        df = pd.read_csv(args.data_path)
    elif os.path.exists("test_data.csv"):
        df = pd.read_csv("test_data.csv")
    else:
        raise FileNotFoundError(
            "Provide --data-path or place test_data.csv in the working directory."
        )

    X, y = make_supervised_table(df, args.features, args.target)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state
    )

    print(f"Dataset: {len(df)} samples, {len(args.features)} features.")
    print(f"Train: {len(X_train)}  Test: {len(X_test)}")

    plotter = DivergenceVisualizer(save_dir=args.save_dir)

    exp_kwargs = dict(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        rounds=args.rounds,
        random_state=args.random_state,
        feature_columns=args.features,
        target_column=args.target,
    )

    if args.instances_diff:
        n_data = args.data_per_instance or (len(X_train) // args.max_instances)
        print(f"\nMode: instances-diff  |  data per instance: {n_data}")
        for n in tqdm.tqdm(
            range(args.min_instances, args.max_instances + 1, args.step_instances)
        ):
            plotter.add_result(
                n,
                do_one_experiment(
                    num_instances=n, sample_per_instance=n_data, **exp_kwargs
                ),
            )
        print_summary(plotter, "Number of Parties")
        plotter.plot("Num Instances", "Number of Instances")

    elif args.data_diff:
        n_inst = args.instances or max(1, len(X_train) // args.max_data)
        print(f"\nMode: data-diff  |  num instances: {n_inst}")
        for n_d in tqdm.tqdm(range(args.min_data, args.max_data + 1, args.step_data)):
            plotter.add_result(
                n_d,
                do_one_experiment(
                    num_instances=n_inst, sample_per_instance=n_d, **exp_kwargs
                ),
            )
        print_summary(plotter, "Data Per Instance")
        plotter.plot("Data Per Instance", "Data Amount")

    elif args.fixed_data:
        print(f"\nMode: fixed-data  |  total data points: {args.total_data_points}")
        for n in tqdm.tqdm(
            range(args.min_instances, args.max_instances + 1, args.step_instances)
        ):
            spp = args.total_data_points // n
            if spp < 1:
                continue
            plotter.add_result(
                n,
                do_one_experiment(
                    num_instances=n, sample_per_instance=spp, **exp_kwargs
                ),
            )
        print_summary(plotter, "Fixed Total Data")
        plotter.plot("Num Instances", "Fixed Data Total")

    else:
        print("No mode selected. Use --instances-diff, --data-diff, or --fixed-data.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Federated learning divergence analysis.")
    p.add_argument("--data-path", type=str)
    p.add_argument("--features", type=str, nargs="+", required=True)
    p.add_argument("--target", type=str, required=True)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="Directory to save plots.  Omit to display interactively.",
    )
    p.add_argument("--random-state", type=int, default=42)

    g_inst = p.add_argument_group("instances-diff mode")
    g_inst.add_argument("--instances-diff", action="store_true")
    g_inst.add_argument("--min-instances", type=int, default=2)
    g_inst.add_argument("--max-instances", type=int, default=10)
    g_inst.add_argument("--step-instances", type=int, default=2)
    g_inst.add_argument("--data-per-instance", type=int, default=None)

    g_data = p.add_argument_group("data-diff mode")
    g_data.add_argument("--data-diff", action="store_true")
    g_data.add_argument("--min-data", type=int, default=10)
    g_data.add_argument("--max-data", type=int, default=100)
    g_data.add_argument("--step-data", type=int, default=10)
    g_data.add_argument("--instances", type=int, default=None)

    g_fixed = p.add_argument_group("fixed-data mode")
    g_fixed.add_argument("--fixed-data", action="store_true")
    g_fixed.add_argument("--total-data-points", type=int, default=1000)

    main(p.parse_args())
