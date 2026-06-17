# A small reproducible study: privacy × robustness × utility in federated learning

Every table below is regenerated from scratch by the `fedbench` CLI under a
fixed seed — no hand-entered numbers:

```bash
fedbench run benchmarks/robustness.toml
fedbench run benchmarks/impossibility.toml
fedbench run benchmarks/privacy.toml
```

The `fed_playground` testbed lets us swap any
(model × aggregation × encryption × attack × partition) and read off the cost.
Three questions, three configs.

## 1. Do the robust aggregators actually resist attacks? (`robustness.toml`)

11 parties, 2 Byzantine, four attacks (one naive sign-flip, two *adaptive*
attacks — IPM and A-Little-Is-Enough — built to defeat distance/median defenses).

See `robustness.md`. FedAvg (`MeanAggregation`) collapses — **MSE 36.4 under
sign-flip, 2.76 under IPM** — while every robust aggregator stays at the clean
optimum (~0.01). Centered clipping is the one that still bleeds a little under
sign-flip/IPM (~0.07), because a fixed clip radius is a blunt instrument; Krum,
Bulyan, median, trimmed-mean, geometric-median and median-of-means all hold.

## 2. Can you have privacy *and* Byzantine robustness at once? (`impossibility.toml`)

This is the interesting one. Masking-based secure aggregation
(`AdditiveSecretSharing`, `PairwiseMaskingEncryption`) hides each party's update
so that only the *sum* is ever revealed. But order/distance defenses (Krum,
median, …) must inspect *individual* updates — which masking has destroyed.

See `impossibility.md`: the masking rows are `—` (incompatible) for Krum and
Median, and only work with `MeanAggregation`. Differential privacy and plaintext
work with everything (they leave per-party values inspectable). This is a genuine
**impossibility frontier**, not a bug — the framework's `is_linear_only` flag
makes the testbed refuse those cells rather than silently compute garbage. If you
want cryptographic input-privacy *and* Byzantine robustness you need heavier
machinery (e.g. MPC-based robust aggregation), which is out of scope here.

## 3. What does differential privacy cost in utility? (`privacy.toml`)

See `privacy.md`. Adding local DP noise raises MSE monotonically: NoEncryption
(0.44) < Gaussian (0.56) < Laplace (4.95) with no attack, and the gap widens
under attack. Laplace's heavier tails (pure ε-DP, no δ) cost more utility than
Gaussian ((ε,δ)-DP) at these settings — the classic privacy/utility trade-off.

## Takeaway

The three axes interact and you cannot maximize all of them: robust aggregation
defeats poisoning but needs plaintext updates; masking gives input-privacy but
only supports linear aggregation; DP gives a tunable privacy knob at a measured
utility cost. `fed_playground` makes each trade-off a one-command experiment.

*Reproducibility: fixed `seed` in every config; attack/DP RNGs seeded explicitly;
leaderboards embed no timestamps, so re-running yields byte-identical tables.*
