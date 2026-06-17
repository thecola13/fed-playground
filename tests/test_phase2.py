"""Tests for Phase 2: config/CLI, leaderboard, load_dataset, seed threading."""

import numpy as np
import pandas as pd
import pytest

from fed_playground.src.benchmark import leaderboard
from fed_playground.src.cli import _classes, _instances, _lookup, run_config
from fed_playground.src.dataloader import DataLoader, load_dataset
from fed_playground.src.environment import Environment


class TestResolution:
    def test_lookup_unknown_raises(self):
        with pytest.raises(ValueError, match="unknown component"):
            _lookup("NotARealClass")

    def test_classes_returns_class_instances_returns_instance(self):
        models = _classes([{"name": "LinearRegressionModel"}])
        assert isinstance(models[0], type)  # the class itself
        aggs = _instances([{"name": "KrumAggregation", "n_byzantine": 3}])
        assert type(aggs[0]).__name__ == "KrumAggregation"
        assert aggs[0].n_byzantine == 3  # params applied


class TestLeaderboard:
    def test_renders_pivot_with_nan_dash(self):
        df = pd.DataFrame(
            [
                {"aggregation": "Mean", "attack": "No", "final_loss": 0.1},
                {"aggregation": "Mean", "attack": "Flip", "final_loss": float("nan")},
                {"aggregation": "Krum", "attack": "No", "final_loss": 0.2},
                {"aggregation": "Krum", "attack": "Flip", "final_loss": 0.3},
            ]
        )
        md = leaderboard(df)
        assert "| aggregation \\ attack |" in md
        assert "—" in md  # NaN cell
        assert "0.100" in md and "0.300" in md
        assert "timestamp" not in md.lower()  # byte-stable, no clock


class TestLoadDataset:
    def test_synthetic_shapes(self):
        loader = load_dataset("synthetic", n_samples=120, n_features=5, seed=1)
        X, y = loader.load()
        assert isinstance(loader, DataLoader)
        assert X.shape == (120, 5) and y.shape == (120,)

    def test_sklearn_offline(self):
        skd = pytest.importorskip("sklearn.datasets")  # skip if extra absent
        assert skd is not None
        loader = load_dataset("sklearn", name="diabetes")
        X, y = loader.load()
        assert X.shape[0] == y.shape[0] > 0

    def test_unknown_kind_raises(self):
        with pytest.raises(ValueError, match="unknown dataset kind"):
            load_dataset("nope")


class TestSeedThreading:
    def test_different_seed_changes_synthetic_data(self):
        a = Environment(n_parties=3, n_features=3, n_samples=100, seed=1)
        b = Environment(n_parties=3, n_features=3, n_samples=100, seed=2)
        a.setup()
        b.setup()
        assert not np.array_equal(a.parties[0].X_train, b.parties[0].X_train)

    def test_same_seed_is_reproducible(self):
        a = Environment(n_parties=3, n_features=3, n_samples=100, seed=7)
        b = Environment(n_parties=3, n_features=3, n_samples=100, seed=7)
        a.setup()
        b.setup()
        np.testing.assert_array_equal(a.parties[0].X_train, b.parties[0].X_train)


class TestCliRunConfig:
    def test_run_config_writes_csv_and_leaderboard(self, tmp_path):
        config = tmp_path / "exp.toml"
        config.write_text(
            """
[experiment]
name = "tiny"
n_parties = 6
n_byzantine = [1]
rounds = 3
seed = 42
[data]
kind = "synthetic"
n_samples = 200
n_features = 3
[grid]
models = [{name="ClosedFormLinearRegressionModel"}]
aggregations = [{name="MeanAggregation"}, {name="KrumAggregation", n_byzantine=1}]
attacks = [{name="NoAttack"}, {name="SignFlipAttack", scale=10}]
[output]
results_csv = "CSV_PATH"
leaderboard_md = "MD_PATH"
""".replace(
                "CSV_PATH", str(tmp_path / "out.csv")
            ).replace(
                "MD_PATH", str(tmp_path / "out.md")
            )
        )
        csv_path = run_config(config)
        assert csv_path.exists()
        df = pd.read_csv(csv_path)
        assert len(df) == 4  # 1 model x 2 agg x 1 enc x 2 attack x 1 nbyz
        assert (tmp_path / "out.md").read_text().strip().startswith("## tiny")
