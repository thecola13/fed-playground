# Phase 2 Spec — Config-driven benchmarks, `fedbench` CLI, leaderboard, study

Status: draft · Date: 2026-06-17 · Depends on: Phase 1 (`run_benchmark`, `Attack`,
`dirichlet_partition`) which is merged and green (152 tests).

## Context & goal

Phase 1 gave us a programmatic benchmark engine (`run_benchmark(...)` → tidy
DataFrame). Phase 2 makes it **reproducible, declarative, and citable** for the
FL-researcher audience: describe an experiment in a TOML file, run it with one
command (`fedbench run …`), and get a committed CSV + an auto-generated
Markdown leaderboard. Capped by a short reproducible **study** that maps the
privacy × robustness × utility frontier — including the empirical
`is_linear_only` impossibility (masking secure-aggregation can't coexist with
order/distance Byzantine defenses).

Out of scope (Phase 3): MkDocs site, notebooks, PyPI/Zenodo release, the
`list-components`/`reproduce` CLI subcommands.

## Guiding constraints (ponytail)
- Core stays pure-NumPy. New deps land only in extras. Prefer stdlib: config =
  `tomllib`; no `tabulate` (a ~15-line Markdown formatter instead).
- Reuse `run_benchmark` and `Environment` wholesale — no second simulation path.
- Component resolution by name = `getattr(fed_playground, name)` validated against
  `__all__`. No registry class.

## Deliverables (decomposed — each its own commit, each ends green)

### Chunk A — config layer + seed reproducibility  (`fed_playground/src/config.py`)
The one real code gap: `Environment` hardcodes the synthetic-data seed (always
42 via `generate_linear_data(self.n_samples, self.n_features)`), so today every
run is identical and `seed` in a config would be a lie.

- Add `seed: int = 42` to `Environment.__init__`; pass it into
  `generate_linear_data(..., random_seed=self.seed)` in `setup`. Add `seed` to
  `run_benchmark(...)` and forward it.
- `config.py`:
  - `resolve(name: str, **params)` → `getattr(fed_playground, name)(**params)` for
    instances, or the class for models; raise a clear `ValueError` if
    `name not in fed_playground.__all__`.
  - `load_config(path) -> dict` via `tomllib`.
  - `build_grid(cfg) -> dict` → kwargs for `run_benchmark` (resolve every grid
    entry; load the dataset, see Chunk D).
- **TOML schema** (grid entries are TOML inline tables → `{name=..., **params}`):
  ```toml
  [experiment]
  name = "robustness"
  n_parties = 11
  rounds = 8
  n_byzantine = [2]
  seed = 42

  [data]
  kind = "synthetic"        # synthetic | sklearn | openml | csv
  n_samples = 900
  n_features = 5

  [grid]
  models      = [{name="ClosedFormLinearRegressionModel"}]
  aggregations = [
    {name="MeanAggregation"},
    {name="KrumAggregation", n_byzantine=2},
    {name="BulyanAggregation", n_byzantine=2},
  ]
  encryptions = [{name="NoEncryption"}]
  attacks     = [
    {name="NoAttack"},
    {name="SignFlipAttack", scale=10},
    {name="IPMAttack", epsilon=2.0},
  ]

  [output]
  results_csv    = "benchmarks/results/robustness.csv"
  leaderboard_md = "benchmarks/RESULTS.md"
  ```
- Tests: `resolve` happy-path + unknown-name error; `load_config` round-trip
  on a fixture TOML; `Environment(seed=1)` vs `seed=2` produce different data.

### Chunk B — `fedbench` CLI  (`fed_playground/src/cli.py`)
- stdlib `argparse`; entry point `[project.scripts] fedbench = "fed_playground.src.cli:main"`.
- Subcommand `run <config.toml>`: `load_config → build_grid → run_benchmark →
  write CSV → write leaderboard`; print the results path. Exit non-zero on a bad
  config.
- Test: invoke `main(["run", tmp_config])` on a tiny 2×2 config; assert the CSV +
  RESULTS.md files appear and are non-empty.

### Chunk C — leaderboard generator  (`fed_playground/src/leaderboard.py`)
- `to_markdown(df) -> str`: ~15-line tidy Markdown table (no tabulate).
- `leaderboard(df, *, index, columns, values="final_loss") -> str`: pivot to an
  attack × defense (or scheme × ε) matrix, round, render, prepend a generated
  header (config name, timestamp passed in — not `datetime.now()`-in-lib so it
  stays deterministic for tests).
- Test: pivot of a hand-built 2×2 DataFrame renders expected Markdown; NaN cells
  show as `—`.

### Chunk D — dataset loaders  (`fed_playground/src/datasets.py`)
- `load_dataset(kind, **opts) -> DataLoader`:
  - `synthetic` → `generate_linear_data` (offline, default).
  - `sklearn` → `load_breast_cancer` / `load_diabetes` (offline, already in the
    `examples` extra). `name` opt selects.
  - `openml` → MNIST via `fetch_openml("mnist_784")` (network, kept per decision).
  - `csv` → existing `DataLoader(file_path=…)`.
- sklearn/openml import lazily; a missing dep raises a clear "install extras"
  message (mirror `example_mnist_federated.py`).
- Test: `synthetic` + `sklearn:breast_cancer` return a `DataLoader` whose `.load()`
  shapes are right. (No network test for openml — assert it's wired, skip the call.)

### Chunk E — the study  (`benchmarks/`)
- 3 committed configs: `robustness.toml` (attack×defense), `privacy_utility.toml`
  (ε sweep over Laplace/Gaussian), `impossibility.toml` (masking × every
  aggregator → the NaN frontier).
- Run them → committed `benchmarks/results/*.csv` + `benchmarks/RESULTS.md`.
- `benchmarks/STUDY.md`: short hand-written narrative interpreting the three
  generated tables (the academic centerpiece; the notebook version is Phase 3).
- Reproduce script note in README: `fedbench run benchmarks/robustness.toml` etc.

## Data flow
`config.toml → tomllib → build_grid (resolve names, load_dataset) → run_benchmark
(Environment per cell, seed-threaded) → DataFrame → {CSV, leaderboard.md}`.
CLI orchestrates; library does the work.

## Verification (per chunk + end-to-end)
- `uv run pytest` green; `ruff`/`black` clean (CI gates).
- New `examples/`/CLI smoke: `fedbench run benchmarks/robustness.toml` regenerates
  `RESULTS.md` **byte-identically** under the fixed seed (reproducibility check).
- `uv build` still succeeds; `fedbench --help` works after `uv sync`.

## Open decisions (confirm before Chunk D/E)
1. Real tabular dataset: `breast_cancer` (classification) and/or `diabetes`
   (regression) from sklearn — offline, zero new data files. OK, or bundle a CSV?
2. Markdown table: hand-rolled formatter (recommended) vs add `tabulate` to deps.
3. Study output location: `benchmarks/` at repo root (recommended) vs under `docs/`.

## Risks
- **Reproducibility leaks**: attack/encryption RNGs are seeded at construction;
  configs must set their seeds for byte-stable `RESULTS.md`. The leaderboard
  header must take an injected timestamp, never read the clock in-library.
- **openml flakiness**: network dataset can't be in CI; keep it out of the gated
  smoke (use synthetic/sklearn there).
- **Scope**: Chunks A–D are mechanical; E (the study + narrative) is where the
  real interpretation lives — budget time there.
