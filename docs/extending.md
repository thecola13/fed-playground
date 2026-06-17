# Extending fedbench

Every concern is an ABC. Subclass one, and the rest of the framework — the
`Environment`, the benchmark engine, the CLI registry — picks it up for free.

## Add an aggregation strategy

Implement one method. It receives the per-party updates and the active
encryption scheme.

```python
import numpy as np
from fed_playground import AggregationStrategy

class TopKMeanAggregation(AggregationStrategy):
    """Average only the k updates closest to the coordinate-wise median."""
    def __init__(self, k: int) -> None:
        self.k = k

    def aggregate(self, encrypted_models, encryption_scheme):
        if encryption_scheme.is_linear_only:        # honour the masking contract
            raise ValueError("needs plaintext updates")
        X = np.stack(encrypted_models)              # (n_parties, n_params)
        med = np.median(X, axis=0)
        order = np.argsort(np.linalg.norm(X - med, axis=1))
        return X[order[: self.k]].mean(axis=0)
```

Use it anywhere a built-in goes:

```python
from fed_playground import Environment
Environment(n_parties=10, n_features=5, n_samples=500,
            aggregation_strategy=TopKMeanAggregation(k=6)).run_simulation(rounds=8)
```

## Add a Byzantine attack

An `Attack` poisons the byzantine parties' updates, given the whole round's
updates (so omniscient attacks can read the honest statistics).

```python
import numpy as np
from fed_playground import Attack

class ConstantAttack(Attack):
    """Byzantine parties all send the same fixed vector."""
    def __init__(self, value: float) -> None:
        self.value = value

    def corrupt(self, updates, byzantine_ids):
        out = list(updates)
        for i in byzantine_ids:
            out[i] = np.full_like(updates[i], self.value)
        return out
```

## Use it in a config

Anything exported from `fed_playground` (i.e. in `__all__`) is addressable by
name in a `fedbench` TOML config:

```toml
[grid]
aggregations = [{ name = "TopKMeanAggregation", k = 6 }]
```

To make a *new* class addressable by name, export it from
`fed_playground/__init__.py` (add it to `__all__`). The benchmark engine and CLI
resolve names via `getattr(fed_playground, name)` — no registry to update.
