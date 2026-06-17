"""`fedbench` command-line interface — run benchmark configs from TOML.

    fedbench run benchmarks/robustness.toml

Reads a TOML experiment config, resolves component names against the public
package API, runs the sweep via :func:`run_benchmark`, and writes a results CSV
plus a Markdown leaderboard. Component params are TOML inline tables, e.g.
``{name="KrumAggregation", n_byzantine=2}``.
"""

from __future__ import annotations

import argparse
import tomllib
from pathlib import Path
from typing import Any

import fed_playground as fp

from .benchmark import leaderboard, run_benchmark
from .dataloader import load_dataset


def _lookup(name: str) -> Any:
    """Resolve a public component by name; error clearly if it is unknown."""
    if name not in fp.__all__:
        raise ValueError(f"unknown component {name!r} (not in fed_playground.__all__)")
    return getattr(fp, name)


def _classes(entries: list[dict]) -> list[Any]:
    return [_lookup(e["name"]) for e in entries]  # models want the class


def _instances(entries: list[dict]) -> list[Any]:
    # aggregations / encryptions / attacks want instances built from their params
    return [
        _lookup(e["name"])(**{k: v for k, v in e.items() if k != "name"})
        for e in entries
    ]


def run_config(path: str | Path) -> Path:
    """Run one TOML config and write CSV + leaderboard; return the CSV path."""
    cfg = tomllib.loads(Path(path).read_text())
    exp = cfg.get("experiment", {})
    grid = cfg.get("grid", {})
    data = cfg.get("data", {"kind": "synthetic"})
    out = cfg.get("output", {})

    kind = data.get("kind", "synthetic")
    loader = None if kind == "synthetic" else load_dataset(**data)

    df = run_benchmark(
        models=_classes(grid.get("models", [{"name": "LinearRegressionModel"}])),
        aggregations=_instances(
            grid.get("aggregations", [{"name": "MeanAggregation"}])
        ),
        encryptions=_instances(grid.get("encryptions", [{"name": "NoEncryption"}])),
        attacks=_instances(grid.get("attacks", [{"name": "NoAttack"}])),
        n_byzantine=tuple(exp.get("n_byzantine", [0])),
        n_parties=exp.get("n_parties", 5),
        rounds=exp.get("rounds", 10),
        seed=exp.get("seed", 42),
        n_features=data.get("n_features", 4),
        n_samples=data.get("n_samples", 500),
        data_loader=loader,
    )

    csv_path = Path(out.get("results_csv", "benchmarks/results.csv"))
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)

    md_path = Path(out.get("leaderboard_md", "benchmarks/RESULTS.md"))
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(
        leaderboard(
            df,
            index=out.get("leaderboard_index", "aggregation"),
            columns=out.get("leaderboard_columns", "attack"),
            title=exp.get("name", "Benchmark"),
        )
    )
    return csv_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="fedbench")
    sub = parser.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run", help="run a benchmark config (TOML)")
    run.add_argument("config", help="path to a .toml experiment config")
    args = parser.parse_args(argv)

    if args.cmd == "run":
        path = run_config(args.config)
        print(f"wrote {path} and its leaderboard")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
