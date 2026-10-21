# Vishwas Multi-Model Fairness & Calibration Comparison Benchmark

> **Evaluation Protocol**: Bootstrap 95% Confidence Intervals (1,000 resamples of lesions).
> **Tone Group Definitions**: Groups defined by `skin_tone_class` values (`G1 = {0,1,2}`, `G2 = {3}`, `G3 = {4,5}`).
> **Headline Macro-F1**: Evaluated across 10 diagnostic classes excluding `MAL_OTH` (extreme dataset sparsity, $n=9$ total).

## 1. Primary Model Comparison Table

| Metric / Dimension | **`d_cw_128`** | **`a_cw_128`** |
| :--- | :---: | :---: |
| **Architecture** | Two-Image + Tabular Fusion | Dermoscopic Only |
| **Mitigation Strategy** | `class_weighted` (cap=10.0) | `class_weighted` (cap=10.0) |
| **Overall Accuracy (95% CI)** | **`47.71%`**<br>[46.30%, 49.07%] | **`45.13%`**<br>[43.87%, 46.55%] |
| **Headline Macro-F1 (95% CI)** *(10 classes)* | **`0.3773`**<br>[0.3611, 0.3938] | **`0.3583`**<br>[0.3423, 0.3754] |
| **Full Macro-F1** *(11 classes)* | `0.3430` | `0.3257` |
| **Balanced Accuracy (95% CI)** | `46.19%`<br>[43.57%, 48.76%] | `43.27%`<br>[41.02%, 45.90%] |
| **Calibration ECE (95% CI)** | `0.0911`<br>[0.0791, 0.1049] | `0.0857`<br>[0.0750, 0.1008] |
| **Macro-AUC (OvR)** | `0.8090` | `0.8030` |
| **--- Subgroup Fairness & Disparity ---** | --- | --- |
| **Worst-Performing Tone Group** | **`G2 = {3}`**<br>($n=3,174$, F1: `0.3611`) | **`G2 = {3}`**<br>($n=3,174$, F1: `0.3363`) |
| **Headline Macro-F1 Disparity Gap** | **`0.0461`** | **`0.0732`** |
| **Balanced Accuracy Disparity Gap** | `3.47%` | `8.87%` |

## 2. Subgroup Performance Across Tone Groups (`skin_tone_class`)

Every group shows sample size $n$, Headline Macro-F1 (95% CI), Balanced Accuracy (95% CI), Accuracy (95% CI), and Calibration ECE (95% CI).

### 2.1 Tone Group: **`G1 = {0,1,2}`**

| Metric | **`d_cw_128`** | **`a_cw_128`** |
| :--- | :---: | :---: |
| **Evaluated Lesions ($n$)** | $n = 617$ | $n = 617$ |
| **Headline Macro-F1 (95% CI)** | **`0.4072`** [0.3512, 0.4534] | **`0.4094`** [0.3550, 0.4587] |
| **Balanced Accuracy (95% CI)** | `45.23%` [38.9%, 52.1%] | `49.53%` [43.6%, 55.7%] |
| **Accuracy (95% CI)** | `51.05%` [47.0%, 54.9%] | `49.92%` [46.0%, 53.6%] |
| **Calibration ECE (95% CI)** | `0.0542` [0.0410, 0.0992] | `0.0633` [0.0455, 0.1063] |

### 2.2 Tone Group: **`G2 = {3}`**

| Metric | **`d_cw_128`** | **`a_cw_128`** |
| :--- | :---: | :---: |
| **Evaluated Lesions ($n$)** | $n = 3,174$ | $n = 3,174$ |
| **Headline Macro-F1 (95% CI)** | **`0.3611`** [0.3386, 0.3809] | **`0.3363`** [0.3146, 0.3580] |
| **Balanced Accuracy (95% CI)** | `46.21%` [42.1%, 50.4%] | `40.66%` [36.8%, 44.9%] |
| **Accuracy (95% CI)** | `47.48%` [45.8%, 49.2%] | `45.02%` [43.2%, 46.9%] |
| **Calibration ECE (95% CI)** | `0.1122` [0.0973, 0.1303] | `0.1080` [0.0912, 0.1257] |

### 2.3 Tone Group: **`G3 = {4,5}`**

| Metric | **`d_cw_128`** | **`a_cw_128`** |
| :--- | :---: | :---: |
| **Evaluated Lesions ($n$)** | $n = 1,449$ | $n = 1,449$ |
| **Headline Macro-F1 (95% CI)** | **`0.3662`** [0.3337, 0.3970] | **`0.3422`** [0.3094, 0.3699] |
| **Balanced Accuracy (95% CI)** | `48.70%` [43.8%, 54.5%] | `46.36%` [41.4%, 52.1%] |
| **Accuracy (95% CI)** | `46.79%` [44.4%, 49.6%] | `43.34%` [40.9%, 45.8%] |
| **Calibration ECE (95% CI)** | `0.0759` [0.0615, 0.1082] | `0.0625` [0.0484, 0.0908] |

## 3. Per-Class Diagnostic Recall Breakdown

| Diagnosis Class | Description | **`d_cw_128` Recall** | **`a_cw_128` Recall** |
| :--- | :--- | :---: | :---: |
| **`AKIEC`** | Actinic Keratosis / Intraepithelial Carcinoma | `54.46%` | `51.49%` |
| **`BCC`** | Basal Cell Carcinoma | `41.20%` | `39.10%` |
| **`BEN_OTH`** | Other Benign Lesions | `34.09%` | `31.82%` |
| **`BKL`** | Benign Keratotic Lesions | `34.74%` | `27.39%` |
| **`DF`** | Dermatofibroma | `50.00%` | `48.08%` |
| **`INF`** | Inflammatory / Infectious Conditions | `40.00%` | `26.00%` |
| `MAL_OTH` *(non-headline)* | Other Malignant Lesions *(non-headline, n=9)* | `0.00%` | `0.00%` |
| **`MEL`** | Melanoma | `46.00%` | `44.89%` |
| **`NV`** | Melanocytic Nevus | `70.51%` | `67.43%` |
| **`SCCKA`** | Squamous Cell Carcinoma / Keratoacanthoma | `58.35%` | `58.99%` |
| **`VASC`** | Vascular Lesions | `78.72%` | `80.85%` |

## 4. Audit & Calibration Artifacts

### Run: `d_cw_128`
- [Fairness Audit Report](file:///Users/priyankachourasia/Downloads/Corrected-Skin-Image-Datasets-main/vishwas/reports/d_cw_128/fairness_audit_report.md)
- [Confusion Matrix Image](file:///Users/priyankachourasia/Downloads/Corrected-Skin-Image-Datasets-main/vishwas/reports/d_cw_128/confusion_matrix.png)
- [Reliability Diagram (Calibration Curves)](file:///Users/priyankachourasia/Downloads/Corrected-Skin-Image-Datasets-main/vishwas/reports/d_cw_128/reliability_curves.png)
- [Predictions CSV](file:///Users/priyankachourasia/Downloads/Corrected-Skin-Image-Datasets-main/vishwas/reports/d_cw_128/predictions.csv)

### Run: `a_cw_128`
- [Fairness Audit Report](file:///Users/priyankachourasia/Downloads/Corrected-Skin-Image-Datasets-main/vishwas/reports/a_cw_128/fairness_audit_report.md)
- [Confusion Matrix Image](file:///Users/priyankachourasia/Downloads/Corrected-Skin-Image-Datasets-main/vishwas/reports/a_cw_128/confusion_matrix.png)
- [Reliability Diagram (Calibration Curves)](file:///Users/priyankachourasia/Downloads/Corrected-Skin-Image-Datasets-main/vishwas/reports/a_cw_128/reliability_curves.png)
- [Predictions CSV](file:///Users/priyankachourasia/Downloads/Corrected-Skin-Image-Datasets-main/vishwas/reports/a_cw_128/predictions.csv)
