# LightGBM Cross Validation Report

## Scope

This report evaluates the existing LightGBM baseline parameters with Stratified 5-Fold Cross Validation. No new features, TF-IDF, BM25, embeddings, or LightGBM parameter changes were introduced.

## Fold Metrics

| fold | train_rows | valid_rows | accuracy | precision | recall | f1 | macro_f1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1.000000 | 391363.000000 | 97841.000000 | 0.882738 | 0.875895 | 0.897740 | 0.886683 | 0.882596 |
| 2.000000 | 391363.000000 | 97841.000000 | 0.882789 | 0.876216 | 0.897420 | 0.886691 | 0.882650 |
| 3.000000 | 391363.000000 | 97841.000000 | 0.882340 | 0.876050 | 0.896620 | 0.886216 | 0.882203 |
| 4.000000 | 391363.000000 | 97841.000000 | 0.883750 | 0.879088 | 0.895720 | 0.887326 | 0.883633 |
| 5.000000 | 391364.000000 | 97840.000000 | 0.881582 | 0.876534 | 0.894240 | 0.885298 | 0.881458 |

## Summary

- Mean Macro F1: 0.882508
- Macro F1 std: 0.000789
- Best fold: 4
- Worst fold: 5
- Baseline validation split Macro F1: 0.884241
- CV mean vs validation split difference: -0.001733
- Runtime: 13.82 seconds

## Confusion Matrices

### Fold 1

| actual \ predicted | 0 | 1 |
| --- | --- | --- |
| 0 | 41481 | 6360 |
| 1 | 5113 | 44887 |

### Fold 2

| actual \ predicted | 0 | 1 |
| --- | --- | --- |
| 0 | 41502 | 6339 |
| 1 | 5129 | 44871 |

### Fold 3

| actual \ predicted | 0 | 1 |
| --- | --- | --- |
| 0 | 41498 | 6343 |
| 1 | 5169 | 44831 |

### Fold 4

| actual \ predicted | 0 | 1 |
| --- | --- | --- |
| 0 | 41681 | 6160 |
| 1 | 5214 | 44786 |

### Fold 5

| actual \ predicted | 0 | 1 |
| --- | --- | --- |
| 0 | 41542 | 6298 |
| 1 | 5288 | 44712 |

## Questions

- Model stabil mi? Model stabil görünüyor; fold Macro F1 standart sapması düşük.
- Overfitting belirtisi var mı? Bu rapor yalnızca validation fold performanslarını ölçer; train skorları hesaplanmadığı için doğrudan overfitting kanıtı yok. Fold skorlarının birbirine yakın olması güçlü bir overfitting sinyali göstermiyor.
- Validation split ile Cross Validation arasında anlamlı fark var mı? Validation split ile Cross Validation arasında anlamlı fark yok.
