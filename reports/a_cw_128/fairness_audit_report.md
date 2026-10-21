# MILK10k Fairness Auditing & Subgroup Evaluation Report

> **Evaluation Engine**: Vishwas Fairness Auditor | **Resamples**: 1,000 (Bootstrap on Lesions)

## 1. Overall Performance Metrics

| Metric | Headline Value | 95% Bootstrap CI | Notes |
| :--- | :---: | :---: | :--- |
| **Overall Accuracy** | `45.13%` | [43.87%, 46.55%] | All evaluated lesions |
| **Headline Macro-F1** | `0.3583` | [0.3423, 0.3754] | **Excludes `MAL_OTH`** (10 diagnostic classes) |
| **Full Macro-F1** | `0.3257` | — | Includes all 11 classes |
| **Balanced Accuracy** | `43.27%` | [41.02%, 45.90%] | Average recall across classes |
| **Macro-AUC (OvR)** | `0.8030` | — | One-vs-Rest ROC-AUC |
| **Calibration (ECE)** | `0.0857` | [0.0750, 0.1008] | Expected Calibration Error |

## 2. Rare Malignancy Audit: `MAL_OTH`

> [!IMPORTANT]
> As required by the evaluation protocol, **`MAL_OTH` is excluded from headline macro metrics** due to extreme dataset sparsity ($n = 9$ lesions in total, 0 lesions in G3). It is audited independently below:

| Rare Diagnosis | Evaluated Lesions ($n$) | Recall | Precision | Subgroup Observations |
| :--- | :---: | :---: | :---: | :--- |
| **`MAL_OTH`** | $n = 9$ | `0.00%` | `0.00%` | Zero representation in G3; 9 samples globally |

## 3. Subgroup Fairness Metrics Across Tone Groups

> **Tone Groups defined by `skin_tone_class` values**: `G1 = {0, 1, 2}`, `G2 = {3}`, `G3 = {4, 5}`.

| Tone Group (`skin_tone_class`) | Lesions ($n$) | Headline Macro-F1 (95% CI) | Balanced Accuracy (95% CI) | Accuracy (95% CI) | Calibration ECE (95% CI) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **`G1`** (`skin_tone_class: {0, 1, 2}`) | $n = 617$ | **`0.4094`** ([0.3550, 0.4587]) | `49.53%` ([43.59%, 55.74%]) | `49.92%` ([46.03%, 53.65%]) | `0.0633` ([0.0455, 0.1063]) |
| **`G2`** (`skin_tone_class: {3}`) | $n = 3,174$ | **`0.3363`** ([0.3146, 0.3580]) | `40.66%` ([36.77%, 44.89%]) | `45.02%` ([43.23%, 46.88%]) | `0.1080` ([0.0912, 0.1257]) |
| **`G3`** (`skin_tone_class: {4, 5}`) | $n = 1,449$ | **`0.3422`** ([0.3094, 0.3699]) | `46.36%` ([41.39%, 52.09%]) | `43.34%` ([40.92%, 45.83%]) | `0.0625` ([0.0484, 0.0908]) |

### 3.1 Subgroup Disparity & Worst-Group Score

- **Worst Performing Group**: **`G2`** ($n = 3,174$, Macro-F1: `0.3363`)
- **Best-Worst Headline Macro-F1 Gap**: **`0.0732`**
- **Balanced Accuracy Gap**: **`8.87%`**

## 4. Per-Class Recall Breakdown Across Groups

| Diagnosis Class | Overall Recall | Overall Precision | G1 Recall | G2 Recall | G3 Recall |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`AKIEC`** | 51.49% | 15.46% | 75.0% | 49.0% | 44.3% |
| **`BCC`** | 39.10% | 87.41% | 7.2% | 42.3% | 35.7% |
| **`BEN_OTH`** | 31.82% | 5.41% | 23.1% | 31.8% | 44.4% |
| **`BKL`** | 27.39% | 27.19% | 36.1% | 25.9% | 25.4% |
| **`DF`** | 48.08% | 12.08% | 76.9% | 40.0% | 35.7% |
| **`INF`** | 26.00% | 6.81% | 40.0% | 25.0% | 20.8% |
| `MAL_OTH` *(non-headline)* | 0.00% | 0.00% | 0.0% | 0.0% | 0.0% |
| **`MEL`** | 44.89% | 47.42% | 51.8% | 43.5% | 38.9% |
| **`NV`** | 67.43% | 71.05% | 68.0% | 65.5% | 69.6% |
| **`SCCKA`** | 58.99% | 45.51% | 66.7% | 60.5% | 55.8% |
| **`VASC`** | 80.85% | 33.33% | 100.0% | 63.6% | 92.9% |

## 5. Sample Size Sparsity Warnings

> [!WARNING]
> The following **5 group x class cells** contained fewer than 10 lesions:
> - Warning: Too few samples for G1 x MAL_OTH (n = 4 lesions < 10)
> - Warning: Too few samples for G1 x SCCKA (n = 6 lesions < 10)
> - Warning: Too few samples for G2 x MAL_OTH (n = 5 lesions < 10)
> - Warning: Too few samples for G3 x BEN_OTH (n = 9 lesions < 10)
> - Warning: Too few samples for G3 x MAL_OTH (n = 0 lesions < 10)
