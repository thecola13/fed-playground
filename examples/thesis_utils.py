"""
Utility functions for master thesis data compatibility.

Provides helper functions to bridge between the master thesis code
and the fed_playground library, enabling seamless data loading and
experiment replication.
"""

import os
import re
import pandas as pd
from typing import List, Dict, Optional, Tuple


def load_thesis_csv(file_path: str, index_col: int = 0) -> pd.DataFrame:
    """
    Load CSV in master thesis format (genes/features as rows).

    Args:
        file_path: Path to CSV file
        index_col: Column to use as row index (default: 0)

    Returns:
        DataFrame with features as rows, samples as columns
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    return pd.read_csv(file_path, index_col=index_col)


def make_supervised_table_compat(
    expr: pd.DataFrame,
    features: List[str],
    target: str,
    transpose: bool = True
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Compatible version of thesis make_supervised_table function.

    Builds X (features) and y (target) with samples as rows.
    Compatible with both fed_playground DataLoader and thesis code.

    Args:
        expr: Expression matrix (genes/features as rows)
        features: List of feature names
        target: Target variable name
        transpose: If True, transpose to samples as rows (default: True)

    Returns:
        Tuple of (X, y) where X is features DataFrame and y is target Series
    """
    if target not in expr.index:
        raise ValueError(f"Target '{target}' not found in data. "
                       f"Available: {list(expr.index)[:10]}...")

    missing_features = [f for f in features if f not in expr.index]
    if missing_features:
        raise ValueError(f"Features {missing_features} not found in data. "
                       f"Available: {list(expr.index)[:10]}...")

    if transpose:
        X = expr.loc[features].T
        y = expr.loc[target]
    else:
        X = expr.loc[features]
        y = expr.loc[target]

    return X, y


def parse_thesis_experiment_name(dirname: str) -> Dict[str, any]:
    """
    Extract experiment parameters from thesis result directory names.

    Parses directory names like:
    - "instances_diff_y_x1-x2-x3-x4_max_15_min_2_samples_8_rounds_20"
    - "data_diff_BRCA1_HMGB2-IL20RA-MCM2-MCM4_max_10_min_2_inst_10_rounds_20"
    - "fixed_data_TP53_RAB17-RRAGA-SMNDC1-SNRPA_total_120_max_15_min_2_rounds_20"

    Args:
        dirname: Directory name to parse

    Returns:
        Dictionary with experiment parameters:
        - mode: "instances_diff", "data_diff", or "fixed_data"
        - target: Target variable name
        - features: List of feature names
        - parameters: Dict of numeric parameters (min, max, samples, rounds, etc.)
    """
    params = {
        "mode": None,
        "target": None,
        "features": [],
        "parameters": {}
    }

    # Extract mode
    if dirname.startswith("instances_diff"):
        params["mode"] = "instances_diff"
    elif dirname.startswith("data_diff"):
        params["mode"] = "data_diff"
    elif dirname.startswith("fixed_data"):
        params["mode"] = "fixed_data"
    else:
        return params

    # Split by underscores
    parts = dirname.split("_")

    # Extract target (3rd element after mode)
    if len(parts) > 2:
        params["target"] = parts[2]

    # Extract features (4th element, split by hyphen)
    if len(parts) > 3:
        params["features"] = parts[3].split("-")

    # Extract numeric parameters
    param_pattern = r"(max|min|samples|inst|instances|rounds|total|step)_(\d+)"
    matches = re.findall(param_pattern, dirname)
    for key, value in matches:
        params["parameters"][key] = int(value)

    return params


def get_thesis_data_path(data_name: str = "synthetic") -> str:
    """
    Get path to master thesis data file.

    Args:
        data_name: Name of dataset ("synthetic", "metabric")

    Returns:
        Full path to data file
    """
    thesis_dir = "/Users/cola/Desktop/master_thesis/data"

    if data_name.lower() == "synthetic":
        return os.path.join(thesis_dir, "synthetic_data_data.csv")
    elif data_name.lower() == "metabric":
        return os.path.join(thesis_dir, "Metabric.csv")
    else:
        raise ValueError(f"Unknown data_name: {data_name}. "
                       f"Choose from: 'synthetic', 'metabric'")


def list_thesis_results(results_dir: str = "/Users/cola/Desktop/master_thesis/graphs") -> List[Dict]:
    """
    List all experiment results from master thesis graphs directory.

    Args:
        results_dir: Path to graphs directory

    Returns:
        List of dictionaries with experiment info
    """
    if not os.path.exists(results_dir):
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    results = []
    for item in os.listdir(results_dir):
        item_path = os.path.join(results_dir, item)
        if os.path.isdir(item_path):
            params = parse_thesis_experiment_name(item)
            if params["mode"]:  # Only include if parsing succeeded
                params["directory"] = item_path
                params["name"] = item
                results.append(params)

    return results


def print_thesis_results_summary(results_dir: str = "/Users/cola/Desktop/master_thesis/graphs"):
    """
    Print a summary of all thesis experiment results.

    Args:
        results_dir: Path to graphs directory
    """
    results = list_thesis_results(results_dir)

    print(f"Found {len(results)} experiment result directories:\n")

    by_mode = {}
    for r in results:
        mode = r["mode"]
        if mode not in by_mode:
            by_mode[mode] = []
        by_mode[mode].append(r)

    for mode, exps in sorted(by_mode.items()):
        print(f"{mode}: {len(exps)} experiments")
        targets = set(e["target"] for e in exps)
        print(f"  Targets: {', '.join(sorted(targets))}")
        print()
