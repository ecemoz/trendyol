# LightGBM Cross Validation Report (with TF-IDF + BM25)

## Scope

This report evaluates the performance stability of the LightGBM classifier with both TF-IDF and BM25 similarity features using Stratified 5-Fold Cross Validation.

## Fold Metrics

| fold | train_rows | valid_rows | accuracy | precision | recall | f1 | macro_f1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1.000000 | 391363.000000 | 97841.000000 | 0.890036 | 0.884708 | 0.902420 | 0.893476 | 0.889921 |
| 2.000000 | 391363.000000 | 97841.000000 | 0.890782 | 0.886416 | 0.901840 | 0.894062 | 0.890677 |
| 3.000000 | 391363.000000 | 97841.000000 | 0.890363 | 0.884419 | 0.903540 | 0.893877 | 0.890243 |
| 4.000000 | 391363.000000 | 97841.000000 | 0.891794 | 0.888543 | 0.901320 | 0.894886 | 0.891700 |
| 5.000000 | 391364.000000 | 97840.000000 | 0.890035 | 0.886847 | 0.899600 | 0.893178 | 0.889939 |

## Summary Table

| Metric | Mean Value |
| --- | --- |
| Accuracy | 0.890602 |
| Precision | 0.886187 |
| Recall | 0.901744 |
| F1 | 0.893896 |
| **Mean Macro F1** | **0.890496** |
| Macro F1 Std | 0.000739 |

- Best fold: 4
- Worst fold: 1
- Runtime: 15.22 seconds

## Progressive CV Comparison

Baseline CV (Macro F1) = 0.882508

↓

TF-IDF CV (Macro F1) = 0.887317 (Değişim: +0.004809)

↓

TF-IDF + BM25 CV (Macro F1) = **0.890496** (Değişim: +0.003179 vs TF-IDF, +0.007988 vs Baseline)

## Short Discussion & Answers

- **Model stabil mi?** Evet, model oldukça stabil görünüyor. Fold'lar arasındaki Macro F1 standard sapması (0.000739) oldukça düşüktür (< 0.001).
- **Overfitting belirtisi var mı?** Hayır, belirgin bir overfitting belirtisi yok. Fold skorları birbirine çok yakın ve dengeli. Validation fold performansı, split validation skoru ile uyumludur.
- **BM25 gerçekten katkı sağlamış mı?** Evet. Sadece TF-IDF içeren modele göre Macro F1 skoru 0.887317 değerinden 0.890496 değerine yükselmiş ve net +0.003179 artış sağlamıştır.
- **TF-IDF + BM25 kombinasyonu kalıcı kullanılmalı mı?** Evet, kesinlikle kalıcı olarak kullanılmalıdır. Modelin hem genelleme skoru en yüksek seviyeye ulaşmış, hem de fold standart sapması düşük kalmaya devam ederek kararlılığını korumuştur.

## Confusion Matrices

### Fold 1

| actual \ predicted | 0 | 1 |
| --- | --- | --- |
| 0 | 41961 | 5880 |
| 1 | 4879 | 45121 |

### Fold 2

| actual \ predicted | 0 | 1 |
| --- | --- | --- |
| 0 | 42063 | 5778 |
| 1 | 4908 | 45092 |

### Fold 3

| actual \ predicted | 0 | 1 |
| --- | --- | --- |
| 0 | 41937 | 5904 |
| 1 | 4823 | 45177 |

### Fold 4

| actual \ predicted | 0 | 1 |
| --- | --- | --- |
| 0 | 42188 | 5653 |
| 1 | 4934 | 45066 |

### Fold 5

| actual \ predicted | 0 | 1 |
| --- | --- | --- |
| 0 | 42101 | 5739 |
| 1 | 5020 | 44980 |

