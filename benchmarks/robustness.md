## Robustness — final global test MSE (lower is better)

| aggregation \ attack | ALittleIsEnoughAttack | IPMAttack | NoAttack | SignFlipAttack |
|---|---|---|---|---|
| BulyanAggregation | 0.011 | 0.010 | 0.010 | 0.010 |
| CenteredClippingAggregation | 0.011 | 0.070 | 0.010 | 0.070 |
| GeometricMedianAggregation | 0.011 | 0.011 | 0.010 | 0.011 |
| KrumAggregation | 0.011 | 0.011 | 0.011 | 0.011 |
| MeanAggregation | 0.011 | 2.756 | 0.010 | 36.393 |
| MedianAggregation | 0.011 | 0.011 | 0.010 | 0.011 |
| MedianOfMeansAggregation | 0.011 | 0.011 | 0.010 | 0.011 |
| TrimmedMeanAggregation | 0.011 | 0.011 | 0.010 | 0.011 |
