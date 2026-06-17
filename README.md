# fed-env

> A modular federated learning simulation framework for research and education.

[![CI](https://github.com/your-org/fed-env/actions/workflows/ci.yml/badge.svg)](https://github.com/your-org/fed-env/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)](pyproject.toml)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

<!-- Badge/URL note: replace `your-org/fed-env` with the real GitHub path before publishing.
     A PyPI badge is added at first release (Phase 3). -->


## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Running Tests](#running-tests)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

**fed-env** is a pure-Python federated learning playground designed to make it
easy to prototype, analyse, and teach the core ideas behind federated learning
(FL) — without requiring a cluster or real network infrastructure.

The library implements a **strategy-pattern architecture**: every key concern
(encryption, aggregation, model type, data loading) is expressed as an abstract
base class that can be swapped out independently.  This makes it straightforward
to answer questions such as "what happens to model divergence as I increase the
number of parties?" or "how does closed-form averaging compare to gradient
descent in a federated setting?"

Out of the box the framework ships with:

- **Models** — gradient descent linear regression, closed-form linear
  regression, L2-regularised ridge regression, logistic regression, and a
  single-hidden-layer MLP (all pure NumPy).
- **Aggregation strategies** — FedAvg (mean), coordinate-wise median, and
  trimmed mean for Byzantine robustness.
- **Encryption schemes** — passthrough baseline, Gaussian differential-privacy
  noise, and additive secret sharing; plus an interface compatible with
  [Concrete ML](https://github.com/zama-ai/concrete-ml) when the optional
  `[fhe]` extra is installed.
- **Four visualizers** — training history curves, model comparison bar charts,
  and divergence analysis plots with per-party breakdowns.

### Research-grounded components

Beyond the baselines above, the framework ships implementations of published
algorithms across every strategy type.  Each class names its source in its
docstring:

| Type | Class | Reference |
|---|---|---|
| Aggregation | `KrumAggregation` | Blanchard et al., *Byzantine-Tolerant Gradient Descent*, NeurIPS 2017 |
| Aggregation | `BulyanAggregation` | El Mhamdi et al., *The Hidden Vulnerability of Distributed Learning in Byzantium*, ICML 2018 |
| Aggregation | `GeometricMedianAggregation` (RFA) | Pillutla et al., *Robust Aggregation for Federated Learning*, IEEE TSP 2022 |
| Aggregation | `MedianOfMeansAggregation` | Nemirovski & Yudin 1983; Lugosi & Mendelson 2019 |
| Aggregation | `CenteredClippingAggregation` | Karimireddy, He & Jaggi, *Learning from History for Byzantine Robust Optimization*, ICML 2021 |
| Model | `SVMModel` | Shalev-Shwartz et al., *Pegasos*, Math. Programming 2011 |
| Model | `LassoRegressionModel` | Tibshirani 1996; Friedman, Hastie & Tibshirani 2010 |
| Model | `ElasticNetRegressionModel` | Zou & Hastie, *Elastic Net*, JRSS-B 2005 |
| Model | `PoissonRegressionModel` | Nelder & Wedderburn, *GLMs*, JRSS-A 1972 |
| Model | `HuberRegressionModel` | Huber, *Robust Estimation of a Location Parameter*, Ann. Math. Stat. 1964 |
| Encryption | `LaplaceDPEncryption` | Dwork et al., *Calibrating Noise to Sensitivity*, TCC 2006 |
| Encryption | `PairwiseMaskingEncryption` | Bonawitz et al., *Practical Secure Aggregation*, ACM CCS 2017 |
| Visualizer | `PrivacyUtilityVisualizer` | Abadi et al., *Deep Learning with Differential Privacy*, ACM CCS 2016 |

Robust aggregators and additive-masking schemes carry an `is_linear_only` flag:
masking schemes (secret sharing, pairwise masks) hide individual updates, so
order/distance-based aggregators (Krum, median, Bulyan, …) correctly refuse
them — only linear aggregation (`MeanAggregation`) is sound over masked shares.

Runnable demos for all of the above live in `examples/` (e.g.
`example_byzantine_robust.py`, `example_svm_lasso.py`,
`example_dp_and_secure_agg.py`, `example_elasticnet_poisson.py`,
`example_bulyan_mom.py`, `example_privacy_utility_curve.py`).

### Benchmarking (`fedbench`)

Describe an experiment in TOML and run the whole sweep with one command:

```bash
fedbench run benchmarks/robustness.toml      # attack × defense matrix
fedbench run benchmarks/impossibility.toml   # privacy × robustness frontier
fedbench run benchmarks/privacy.toml         # DP utility cost
```

Each run writes a results CSV and a Markdown leaderboard (NaN cells = incompatible
combinations, e.g. masking secure-aggregation × order-statistic defense). The
committed [`benchmarks/STUDY.md`](benchmarks/STUDY.md) interprets the three —
reproducible byte-for-byte under the configs' fixed seeds.

### Optional: Fully Homomorphic Encryption (FHE)

The `[fhe]` optional extra installs **Concrete ML** (Zama's FHE library).
This requires **Python 3.11 or 3.12** (concrete-ml does not yet support 3.13+).
When installed, custom `EncryptionScheme` subclasses can perform genuine
ciphertext-level operations during aggregation.  Without it the framework
operates entirely in plaintext — which is correct for educational purposes.

---

## Project Structure

```
fed_env/
├── fed_playground/             # Installable Python package
│   ├── __init__.py             # Public API — import everything from here
│   └── src/
│       ├── aggregation.py      # AggregationStrategy ABC + MeanAggregation
│       ├── dataloader.py       # DataLoader: CSV / DataFrame / numpy inputs
│       ├── encryption.py       # EncryptionScheme ABC + NoEncryption baseline
│       ├── environment.py      # Environment: high-level simulation driver
│       ├── models.py           # Model ABC + Linear / ClosedForm implementations
│       ├── orchestrator.py     # Orchestrator: broadcast + aggregate
│       ├── party.py            # Party: local training + encrypted updates
│       ├── utils_data.py       # Synthetic data generation and splitting
│       └── visualization.py   # TrainingHistory / Comparison / Divergence plots
├── examples/
│   ├── basic_simulation.py     # Minimal end-to-end FL demo
│   ├── visualization_demo.py   # All three visualizers in one script
│   ├── divergence_analysis.py  # CLI tool for divergence experiments
│   └── test_data.csv           # Bundled 200-sample toy dataset
├── tests/                      # pytest test suite (mirrors src/ layout)
├── .env.example                # Template for environment variables
├── pyproject.toml              # Project metadata, dependencies, tool config
└── LICENSE                     # MIT
```

---

## Requirements

- **Python** ≥ 3.11, < 3.13
- **System**: no compiled extensions required for the core library
- **Optional FHE**: `concrete-ml ≥ 1.9.0` (Python 3.11 or 3.12 only, macOS/Linux)

Core Python dependencies (installed automatically):

| Package | Purpose |
|---|---|
| `numpy ≥ 1.24` | Numerical computing |
| `pandas ≥ 2.0` | Data loading and manipulation |
| `matplotlib ≥ 3.7` | Visualizations |
| `tqdm ≥ 4.65` | Progress bars in example scripts |

---

## Installation

### Using uv (recommended)

[uv](https://docs.astral.sh/uv/) is the fastest way to get started — it manages
the Python version and virtual environment automatically.

```bash
# 1. Clone the repository
git clone https://github.com/your-org/fed-env.git
cd fed-env

# 2. Install dependencies and create the virtual environment
uv sync

# 3. (Optional) Include development tools
uv sync --extra dev

# 4. (Optional) Include FHE support (Python 3.11 or 3.12 required)
uv sync --extra fhe
```

Run scripts directly without activating the environment:

```bash
uv run python examples/basic_simulation.py --parties 4 --rounds 10
uv run pytest
```

Or activate the environment first:

```bash
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python examples/basic_simulation.py
pytest
```

### Using pip (alternative)

```bash
# 1. Clone the repository
git clone https://github.com/your-org/fed-env.git
cd fed-env

# 2. Create and activate a virtual environment
python3.11 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. Install the library in editable mode
pip install -e .

# 4. (Optional) Install development tools
pip install -e ".[dev]"

# 5. (Optional) FHE support
pip install -e ".[fhe]"
```

Verify the installation:

```bash
python -c "import fed_playground; print('fed-env installed successfully')"
```

---

## Configuration

The library reads a small number of optional environment variables.
Copy `.env.example` to `.env` and edit as needed — the file is git-ignored.

| Variable | Required | Default | Description |
|---|---|---|---|
| `THESIS_DATA_DIR` | No | — | Path to an external research data directory. Used by private example scripts only; not needed for the published examples. |
| `FED_LOG_LEVEL` | No | `WARNING` | Python logging level for the `fed_playground` logger (`DEBUG`, `INFO`, `WARNING`, `ERROR`). |

The library uses Python's standard `logging` module.  To see round-by-round
progress, set `FED_LOG_LEVEL=INFO` or configure logging in your script:

```python
import logging
logging.basicConfig(level=logging.INFO)
```

---

## Usage

### Basic simulation

```python
from fed_playground import (
    Environment,
    NoEncryption,
    MeanAggregation,
    ClosedFormLinearRegressionModel,
    DataLoader,
)

# Load data from a CSV file
loader = DataLoader(file_path="examples/test_data.csv", target_column="target")

# Build and run a 3-party federated simulation for 10 rounds
env = Environment(
    n_parties=3,
    encryption_scheme=NoEncryption(),
    aggregation_strategy=MeanAggregation(),
    model_class=ClosedFormLinearRegressionModel,
    data_loader=loader,
)

history = env.run_simulation(rounds=10)

print(history["global_loss"])   # per-round MSE on the held-out test set
print(history["party_loss"])    # per-round average local training MSE
```

### Using synthetic data

```python
env = Environment(
    n_parties=5,
    n_features=10,
    n_samples=500,
    encryption_scheme=NoEncryption(),
    aggregation_strategy=MeanAggregation(),
    model_class=ClosedFormLinearRegressionModel,
)
history = env.run_simulation(rounds=20)
```

### Visualizing results

```python
from fed_playground import TrainingHistoryVisualizer, ComparisonVisualizer

# Plot training curves (displayed interactively)
viz = TrainingHistoryVisualizer()
viz.plot(
    data={"Global Loss": history["global_loss"], "Party Loss": history["party_loss"]},
    title="Federated Training",
    xlabel="Round",
    ylabel="MSE",
)

# Save to a file instead
viz_save = TrainingHistoryVisualizer(save_dir="./plots")
viz_save.plot(data=..., filename="training.png")
```

### Extending with a custom model

```python
import numpy as np
from fed_playground import Model, Environment, NoEncryption, MeanAggregation

class RidgeRegressionModel(Model):
    def __init__(self, input_dim: int, alpha: float = 0.01) -> None:
        self.input_dim = input_dim
        self.alpha = alpha
        self.params = np.zeros(input_dim + 1)

    def train(self, X, y):
        n = X.shape[0]
        X_b = np.hstack([X, np.ones((n, 1))])
        I = np.eye(X_b.shape[1])
        I[-1, -1] = 0  # do not regularize bias term
        self.params = np.linalg.solve(X_b.T @ X_b + self.alpha * I, X_b.T @ y)

    def get_parameters(self): return self.params
    def set_parameters(self, p): self.params = p
    def predict(self, X): return np.hstack([X, np.ones((X.shape[0], 1))]) @ self.params
    def evaluate(self, X, y): return float(np.mean((y - self.predict(X)) ** 2))

env = Environment(
    n_parties=4,
    n_features=5,
    n_samples=200,
    model_class=RidgeRegressionModel,
    model_params={"alpha": 0.1},
)
env.run_simulation(rounds=5)
```

### CLI example scripts

```bash
# Basic simulation with a table of per-round metrics
python examples/basic_simulation.py --parties 4 --rounds 10

# Divergence analysis: vary party count, display plots interactively
python examples/divergence_analysis.py \
    --data-path examples/test_data.csv \
    --features feature_1 feature_2 feature_3 feature_4 feature_5 \
    --target target \
    --instances-diff --min-instances 2 --max-instances 8

# Save plots to disk instead of showing them
python examples/divergence_analysis.py \
    --data-path examples/test_data.csv \
    --features feature_1 feature_2 feature_3 feature_4 feature_5 \
    --target target \
    --instances-diff \
    --save-dir ./results
```

---

## Running Tests

```bash
# Run the full suite
uv run pytest

# Run a specific module
uv run pytest tests/test_environment.py

# Run with coverage report
uv run pytest --cov=fed_playground --cov-report=term-missing
```

A passing run looks like:

```
tests/test_aggregation.py    .....
tests/test_dataloader.py     ............
tests/test_encryption.py     ......
tests/test_environment.py    .......
tests/test_models.py         ...........
tests/test_orchestrator.py   .....
tests/test_party.py          .......
tests/test_utils_data.py     .......
```

---

## Contributing

### Setting up a dev environment

```bash
git clone https://github.com/your-org/fed-env.git
cd fed-env
uv sync --extra dev
```

### Branch naming

| Type | Prefix | Example |
|---|---|---|
| New feature | `feat/` | `feat/add-dp-noise` |
| Bug fix | `fix/` | `fix/dataloader-elif` |
| Documentation | `docs/` | `docs/update-readme` |
| Refactoring | `refactor/` | `refactor/split-orchestrator` |

### Commit style

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
feat(aggregation): add weighted FedAvg strategy
fix(dataloader): correct elif branch for DataFrame input
docs(readme): add custom model example
```

### Pull request checklist

- [ ] `pytest` passes with zero failures
- [ ] `ruff check .` reports zero violations
- [ ] `black --check .` reports no formatting issues
- [ ] New public functions/classes have Google-style docstrings
- [ ] New behaviour is covered by at least one test

---

## License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for
the full text.
