# MILK10k Fairness Auditing & Subgroup Evaluation Report

> **Evaluation Engine**: Vishwas Fairness Auditor | **Resamples**: 1,000 (Bootstrap on Lesions)

## 1. Overall Performance Metrics

| Metric | Headline Value | 95% Bootstrap CI | Notes |
| :--- | :---: | :---: | :--- |
| **Overall Accuracy** | `66.03%` | [63.45%, 69.08%] | All evaluated lesions |
| **Headline Macro-F1** | `0.3569` | [0.3114, 0.3945] | **Excludes `MAL_OTH`** (10 diagnostic classes) |
| **Full Macro-F1** | `0.3245` | — | Includes all 11 classes |
| **Balanced Accuracy** | `32.28%` | [28.21%, 37.87%] | Average recall across classes |
| **Macro-AUC (OvR)** | `0.8444` | — | One-vs-Rest ROC-AUC |
| **Calibration (ECE)** | `0.0528` | — | Expected Calibration Error |

## 2. Rare Malignancy Audit: `MAL_OTH`

> [!IMPORTANT]
> As required by the evaluation protocol, **`MAL_OTH` is excluded from headline macro metrics** due to extreme dataset sparsity ($n = 2$ lesions in total, 0 lesions in G3). It is audited independently below:

| Rare Diagnosis | Evaluated Lesions ($n$) | Recall | Precision | Subgroup Observations |
| :--- | :---: | :---: | :---: | :--- |
| **`MAL_OTH`** | 2 | `0.00%` | `0.00%` | Zero representation in G3; 9 samples globally |

## 3. Subgroup Fairness Metrics Across Tone Groups

Performance broken down by tone group `[G1, G2, G3]`:

| Tone Group | Lesions ($n$) | Headline Macro-F1 (95% CI) | Balanced Accuracy (95% CI) | Accuracy | ECE |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **`G1`** | 115 | **`0.3472`** ([0.2557, 0.4174]) | `35.86%` ([29.10%, 52.80%]) | `58.26%` | `0.1198` |
| **`G2`** | 655 | **`0.3408`** ([0.2895, 0.3939]) | `30.77%` ([26.36%, 40.02%]) | `68.70%` | `0.0434` |
| **`G3`** | 278 | **`0.3593`** ([0.2242, 0.3907]) | `35.18%` ([25.41%, 41.46%]) | `62.95%` | `0.0792` |

### 3.1 Subgroup Disparity & Worst-Group Score

- **Worst Performing Group**: **`G2`** (Macro-F1: `0.3408`)
- **Best-Worst Macro-F1 Gap**: **`0.0185`**
- **Balanced Accuracy Gap**: **`5.10%`**

## 4. Per-Class Recall Breakdown Across Groups

| Diagnosis Class | Overall Recall | Overall Precision | G1 Recall | G2 Recall | G3 Recall |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`AKIEC`** | 26.09% | 40.91% | 63.6% | 20.5% | 14.3% |
| **`BCC`** | 88.71% | 75.42% | 66.7% | 89.1% | 90.2% |
| **`BEN_OTH`** | 0.00% | 0.00% | 0.0% | 0.0% | 0.0% |
| **`BKL`** | 33.65% | 31.82% | 57.1% | 32.4% | 22.7% |
| **`DF`** | 0.00% | 0.00% | 0.0% | 0.0% | 0.0% |
| **`INF`** | 0.00% | 0.00% | 0.0% | 0.0% | 0.0% |
| `MAL_OTH` *(non-headline)* | 0.00% | 0.00% | 0.0% | 0.0% | 0.0% |
| **`MEL`** | 27.06% | 65.71% | 35.5% | 23.5% | 20.0% |
| **`NV`** | 80.56% | 67.84% | 85.7% | 80.0% | 77.3% |
| **`SCCKA`** | 49.00% | 56.98% | 0.0% | 59.7% | 27.3% |
| **`VASC`** | 50.00% | 42.86% | 50.0% | 33.3% | 100.0% |

## 5. Sample Size Sparsity Warnings

> [!WARNING]
> The following **16 group x class cells** contained fewer than 10 lesions:
> - Warning: Too few samples for G1 x BEN_OTH (n = 2 lesions < 10)
> - Warning: Too few samples for G1 x DF (n = 1 lesions < 10)
> - Warning: Too few samples for G1 x INF (n = 3 lesions < 10)
> - Warning: Too few samples for G1 x MAL_OTH (n = 1 lesions < 10)
> - Warning: Too few samples for G1 x SCCKA (n = 0 lesions < 10)
> - Warning: Too few samples for G1 x VASC (n = 2 lesions < 10)
> - Warning: Too few samples for G2 x BEN_OTH (n = 5 lesions < 10)
> - Warning: Too few samples for G2 x DF (n = 5 lesions < 10)
> - Warning: Too few samples for G2 x INF (n = 6 lesions < 10)
> - Warning: Too few samples for G2 x MAL_OTH (n = 1 lesions < 10)
> - Warning: Too few samples for G2 x VASC (n = 3 lesions < 10)
> - Warning: Too few samples for G3 x BEN_OTH (n = 1 lesions < 10)
> - Warning: Too few samples for G3 x DF (n = 3 lesions < 10)
> - Warning: Too few samples for G3 x INF (n = 7 lesions < 10)
> - Warning: Too few samples for G3 x MAL_OTH (n = 0 lesions < 10)
> - Warning: Too few samples for G3 x VASC (n = 1 lesions < 10)
