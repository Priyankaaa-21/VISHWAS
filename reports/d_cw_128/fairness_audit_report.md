# MILK10k Fairness Auditing & Subgroup Evaluation Report

> **Evaluation Engine**: Vishwas Fairness Auditor | **Resamples**: 1,000 (Bootstrap on Lesions)

## 1. Overall Performance Metrics

| Metric | Headline Value | 95% Bootstrap CI | Notes |
| :--- | :---: | :---: | :--- |
| **Overall Accuracy** | `47.71%` | [46.30%, 49.07%] | All evaluated lesions |
| **Headline Macro-F1** | `0.3773` | [0.3611, 0.3938] | **Excludes `MAL_OTH`** (10 diagnostic classes) |
| **Full Macro-F1** | `0.3430` | — | Includes all 11 classes |
| **Balanced Accuracy** | `46.19%` | [43.57%, 48.76%] | Average recall across classes |
| **Macro-AUC (OvR)** | `0.8090` | — | One-vs-Rest ROC-AUC |
| **Calibration (ECE)** | `0.0911` | [0.0791, 0.1049] | Expected Calibration Error |

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
| **`G1`** (`skin_tone_class: {0, 1, 2}`) | $n = 617$ | **`0.4072`** ([0.3512, 0.4534]) | `45.23%` ([38.87%, 52.15%]) | `51.05%` ([47.00%, 54.95%]) | `0.0542` ([0.0410, 0.0992]) |
| **`G2`** (`skin_tone_class: {3}`) | $n = 3,174$ | **`0.3611`** ([0.3386, 0.3809]) | `46.21%` ([42.11%, 50.39%]) | `47.48%` ([45.84%, 49.21%]) | `0.1122` ([0.0973, 0.1303]) |
| **`G3`** (`skin_tone_class: {4, 5}`) | $n = 1,449$ | **`0.3662`** ([0.3337, 0.3970]) | `48.70%` ([43.82%, 54.52%]) | `46.79%` ([44.37%, 49.62%]) | `0.0759` ([0.0615, 0.1082]) |

### 3.1 Subgroup Disparity & Worst-Group Score

- **Worst Performing Group**: **`G2`** ($n = 3,174$, Macro-F1: `0.3611`)
- **Best-Worst Headline Macro-F1 Gap**: **`0.0461`**
- **Balanced Accuracy Gap**: **`3.47%`**

## 4. Per-Class Recall Breakdown Across Groups

| Diagnosis Class | Overall Recall | Overall Precision | G1 Recall | G2 Recall | G3 Recall |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`AKIEC`** | 54.46% | 20.45% | 72.5% | 51.5% | 52.5% |
| **`BCC`** | 41.20% | 87.61% | 10.3% | 43.1% | 40.9% |
| **`BEN_OTH`** | 34.09% | 4.42% | 15.4% | 50.0% | 22.2% |
| **`BKL`** | 34.74% | 30.43% | 36.1% | 35.9% | 30.5% |
| **`DF`** | 50.00% | 14.21% | 76.9% | 40.0% | 42.9% |
| **`INF`** | 40.00% | 8.16% | 30.0% | 43.8% | 41.7% |
| `MAL_OTH` *(non-headline)* | 0.00% | 0.00% | 0.0% | 0.0% | 0.0% |
| **`MEL`** | 46.00% | 52.54% | 51.1% | 45.5% | 40.7% |
| **`NV`** | 70.51% | 74.08% | 73.4% | 69.0% | 70.1% |
| **`SCCKA`** | 58.35% | 48.34% | 50.0% | 61.5% | 52.8% |
| **`VASC`** | 78.72% | 28.03% | 81.8% | 68.2% | 92.9% |

## 5. Sample Size Sparsity Warnings

> [!WARNING]
> The following **5 group x class cells** contained fewer than 10 lesions:
> - Warning: Too few samples for G1 x MAL_OTH (n = 4 lesions < 10)
> - Warning: Too few samples for G1 x SCCKA (n = 6 lesions < 10)
> - Warning: Too few samples for G2 x MAL_OTH (n = 5 lesions < 10)
> - Warning: Too few samples for G3 x BEN_OTH (n = 9 lesions < 10)
> - Warning: Too few samples for G3 x MAL_OTH (n = 0 lesions < 10)
