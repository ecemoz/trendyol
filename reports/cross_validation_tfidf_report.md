# LightGBM Cross Validation Report (with TF-IDF)

## Scope

This report evaluates the performance stability of the LightGBM classifier with TF-IDF similarity features using Stratified 5-Fold Cross Validation.

## Fold Metrics

| fold | train_rows | valid_rows | accuracy | precision | recall | f1 | macro_f1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1.000000 | 391363.000000 | 97841.000000 | 0.888053 | 0.881251 | 0.902560 | 0.891778 | 0.887920 |
| 2.000000 | 391363.000000 | 97841.000000 | 0.887716 | 0.882190 | 0.900540 | 0.891271 | 0.887596 |
| 3.000000 | 391363.000000 | 97841.000000 | 0.886602 | 0.879406 | 0.901760 | 0.890442 | 0.886462 |
| 4.000000 | 391363.000000 | 97841.000000 | 0.888309 | 0.884143 | 0.899280 | 0.891647 | 0.888202 |
| 5.000000 | 391364.000000 | 97840.000000 | 0.886519 | 0.881890 | 0.898240 | 0.889990 | 0.886406 |

## Summary & Comparison

| Metric | Baseline CV | With TF-IDF CV | Change |
| --- | --- | --- | --- |
| Mean Macro F1 | 0.882508 | 0.887317 | +0.004809 |
| Macro F1 Std | 0.000789 | 0.000835 | +0.000046 |

- Best fold: 4
- Worst fold: 5
- Runtime: 14.54 seconds

## Confusion Matrices

### Fold 1

| actual \ predicted | 0 | 1 |
| --- | --- | --- |
| 0 | 41760 | 6081 |
| 1 | 4872 | 45128 |

### Fold 2

| actual \ predicted | 0 | 1 |
| --- | --- | --- |
| 0 | 41828 | 6013 |
| 1 | 4973 | 45027 |

### Fold 3

| actual \ predicted | 0 | 1 |
| --- | --- | --- |
| 0 | 41658 | 6183 |
| 1 | 4912 | 45088 |

### Fold 4

| actual \ predicted | 0 | 1 |
| --- | --- | --- |
| 0 | 41949 | 5892 |
| 1 | 5036 | 44964 |

### Fold 5

| actual \ predicted | 0 | 1 |
| --- | --- | --- |
| 0 | 41825 | 6015 |
| 1 | 5088 | 44912 |

