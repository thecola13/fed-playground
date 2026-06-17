# fedbench

A modular, **pure-NumPy** federated learning simulation and benchmark framework
for research and education. Swap any
(model × aggregation × encryption × attack × partition) and read off the
privacy / robustness / utility trade-off — no cluster, no heavy deps.

## Install

```bash
uv sync                 # core
uv sync --extra examples  # + scikit-learn, tqdm (some demos / datasets)
pip install fedbench      # once released
```

## 30-second tour

```python
from fed_playground import Environment, KrumAggregation, SignFlipAttack

env = Environment(
    n_parties=11, n_features=5, n_samples=900,
    aggregation_strategy=KrumAggregation(n_byzantine=2),
    attack=SignFlipAttack(scale=10), n_byzantine=2,
)
print(env.run_simulation(rounds=8)["global_loss"][-1])  # robust under attack
```

Or sweep a whole matrix with one command:

```bash
fedbench run benchmarks/robustness.toml
```

## Where to go next

- **[API reference](api.md)** — every public class, pulled from its docstring.
- **[Extending](extending.md)** — add your own aggregator or attack in ~20 lines.
- **[Benchmark study](study.md)** — the reproducible privacy × robustness × utility results.

The framework is built on a strategy pattern: `Model`, `AggregationStrategy`,
`EncryptionScheme`, `Attack`, and `Visualizer` are independent, swappable ABCs.
Robust aggregators and masking encryption schemes carry an `is_linear_only`
contract so incompatible combinations are refused, not silently miscomputed.
