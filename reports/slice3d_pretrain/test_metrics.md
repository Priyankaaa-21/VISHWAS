# SLICE-3D Stage-1 Pretraining: Test Evaluation Metrics

> **Evaluation Protocol**: Single evaluation on the held-out test split (25,000 images).
> **Confidence Intervals**: 1,000 bootstrap resamples clustered at the patient level.
> **Metric Standard**: Diagnostic accuracy is omitted; ranking and sensitivity metrics are reported.

## 1. Primary Test Metrics

| Evaluation Metric | Point Estimate | 95% Bootstrap CI (Patient-Clustered) | Clinical Definition |
| :--- | :---: | :---: | :--- |
| **pAUC (TPR $\ge 0.80$)** | **`0.1045`** | **`[0.0721, 0.1330]`** | Official ISIC 2024 metric (high-sensitivity partial AUC) |
| **ROC-AUC** | **`0.8664`** | **`[0.8072, 0.9043]`** | Full receiver operating characteristic area |
| **Test Prevalence** | **`7.7600%`** | --- | `97` malignant cases / `1250` total |

## 2. Test Cohort Characteristics

| Dimension | Count | Note |
| :--- | :---: | :--- |
| **Evaluated Images** | `1,250` | Fixed test partition |
| **Evaluated Patients** | `226` | Zero overlap with training/validation cohorts |
| **Malignant Samples ($y=1$)** | `97` | All positive cases preserved in partition |
| **Benign Samples ($y=0$)** | `1,153` | Subsampled benign majority |

> **Prevalence Notice**: Test prevalence is 0.3880%, enriched relative to the 0.0980% raw dataset base rate. Both ROC-AUC and pAUC are insensitive to prevalence shifts, providing robust benchmark comparisons.