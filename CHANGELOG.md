# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **`fedbench` CLI** (`fedbench run config.toml`): declarative TOML experiments →
  results CSV + auto-generated Markdown leaderboard. Component params are TOML
  inline tables resolved against the public API.
- **`load_dataset`**: synthetic / sklearn (offline) / openml-MNIST / csv data
  sources for benchmarks.
- **`leaderboard`**: render a results DataFrame as a Markdown pivot (NaN→"—"),
  no `tabulate` dep, no embedded timestamp (byte-stable).
- **`seed`** threaded through `Environment`/`run_benchmark` for reproducible runs.
- **`benchmarks/`**: reproducible study (robustness, privacy×robustness
  impossibility, DP utility-cost) with committed configs, leaderboards, and
  `STUDY.md` narrative.
- **Benchmark engine** (`run_benchmark`): one-call Cartesian sweep over
  (model × aggregation × encryption × attack × n_byzantine) → tidy DataFrame;
  incompatible privacy×robustness cells recorded as NaN, not crashes.
- **`Attack` strategy type** (`attacks.py`): `NoAttack`, `SignFlipAttack`,
  `GaussianAttack`, `IPMAttack` (Xie et al. 2020), `ALittleIsEnoughAttack`
  (Baruch et al. 2019). Wired into `Orchestrator`/`Environment` via
  `attack` + `n_byzantine` (replaces ad-hoc Byzantine `Party` subclassing).
- **`dirichlet_partition`**: non-IID label-skew data partitioning (Hsu et al. 2019).
- Byzantine-robust aggregation strategies: `KrumAggregation` (incl. Multi-Krum),
  `BulyanAggregation`, `GeometricMedianAggregation` (RFA), `MedianOfMeansAggregation`,
  `CenteredClippingAggregation`.
- Research-grounded models: `SVMModel` (Pegasos), `LassoRegressionModel`,
  `ElasticNetRegressionModel`, `PoissonRegressionModel`, `HuberRegressionModel`.
- Privacy schemes: `LaplaceDPEncryption` (ε-DP), `PairwiseMaskingEncryption`
  (Bonawitz secure aggregation); finished `AdditiveSecretSharing` with real
  mask cancellation.
- `PrivacyUtilityVisualizer` for DP privacy/utility trade-off curves.
- `is_linear_only` contract: order/distance-based aggregators reject
  additive-masking encryption schemes instead of silently miscomputing.
- Runnable, tested example scripts for every new component.
- Packaging metadata (classifiers, keywords, URLs, `py.typed`), `CITATION.cff`,
  GitHub Actions CI, and README badges.

### Changed
- `Model.evaluate` is now a concrete MSE default on the ABC (classifiers/GLMs override).
- `tqdm` moved from core dependencies to the `[examples]` extra.

## [0.1.0]

### Added
- Initial release: strategy-pattern federated learning simulator (`Environment`,
  `Orchestrator`, `Party`) with linear/ridge/closed-form/logistic/MLP models,
  FedAvg/median/trimmed-mean aggregation, NoEncryption/Gaussian-DP/secret-sharing
  schemes, data loading, and visualizers.
