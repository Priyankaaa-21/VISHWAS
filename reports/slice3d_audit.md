# SLICE-3D (ISIC 2024) Dataset Audit & Integrity Report

> **Audit Date**: 2026-11-15 | **Audit Runtime**: 5.96s | **Engine**: Vishwas SLICE-3D Auditor

## 1. Dataset Overview & Source Metadata

The **SLICE-3D (ISIC 2024 Challenge)** dataset is a large-scale dermatological repository comprising 3D Total Body Photography (3D-TBP) lesion image crops.

| Attribute | Value | Note |
| :--- | :---: | :--- |
| **Total Metadata Rows** | **`401,059`** | Joined on `isic_id` |
| **Total Image Files on Disk** | **`401,059`** | Verified via `os.scandir` |
| **Missing Images** | **`0`** | 100% matched across metadata and disk |
| **Corrupt Images (Sample n=2,000)** | **`0`** | PIL verification pass |
| **Unique Patients** | **`1,042`** | Column `patient_id` |
| **Patients with Malignant Lesions** | **`259`** | 24.9% of patient cohort |

## 2. Diagnostic Target Distribution & Severe Class Imbalance

| Diagnostic Label | Image Count | Prevalence (%) | Clinical Implication |
| :--- | :---: | :---: | :--- |
| **Benign (`malignant = 0.0`)** | `400,666` | `99.9020%` | Dominant majority class |
| **Malignant (`malignant = 1.0`)** | `393` | `0.0980%` | Critical minority class |
| **Total Lesions** | `401,059` | `100.00%` | --- |

> **Class Imbalance Ratio**: **`1019.5 : 1`** (approximately 1,020 benign lesions per 1 malignant lesion).
>
> **Methodological Implication**: Standard accuracy is clinically uninformative and highly misleading in this regime (a naive model predicting all benign achieves 99.90% accuracy). Stage-1 pretraining and evaluation must employ partial AUC (pAUC above 80% TPR), ROC-AUC, and class-weighted / balanced sampling.

## 3. Patient Demographic & Clinical Metadata Completeness

| Metadata Field | Column Found | Non-Null Count | Missing Count | Missingness (%) |
| :--- | :--- | :---: | :---: | :---: |
| **Patient ID** | `patient_id` | `401,059` | `0` | `0.00%` |
| **Approximate Age** | `age_approx` | `398,261` | `2,798` | `0.70%` |
| **Patient Sex** | `sex` | `389,542` | `11,517` | `2.87%` |
| **Anatomical Site** | `anatom_site_general` | `395,303` | `5,756` | `1.44%` |

> **Note on Skin Tone**: SLICE-3D contains **no ground truth skin-tone labels** (Fitzpatrick / Individual Typology Angle). Fairness auditing is strictly reserved for the MILK10k cohort where confirmed `skin_tone_class` labels exist.

## 4. Image Dimensions & Resolution Distribution (Sample $n=2,000$)

All crops are standardized RGB clinical 3D-TBP patches extracted by the ISIC 2024 pipeline.

| Dimension Metric | Minimum | 25th %ile | Median | Mean | 75th %ile | Maximum |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Width (px)** | `61` | `121` | `131` | `133.0` | `143` | `233` |
| **Height (px)** | `61` | `121` | `131` | `133.0` | `143` | `233` |
| **File Size (KB)** | `0.9` | --- | `2.7` | `2.8` | --- | `6.1` |

## 5. Audit Conclusions & Subsetting Strategy

1. **Full Integrity Confirmed**: Zero missing image files and zero corrupt images detected.
2. **Extreme Class Imbalance**: With only 393 malignant images across 401,059 total images, any random subsampling without stratification would decimate positive cases.
3. **Curated 100,000-Image Stage-1 Cohort**: As prescribed by the project specification, a fixed subset of 75,000 training images (including 7,500 validation) and 25,000 test images will be extracted:
   - **Strict Patient Grouping**: All crops from any single patient reside exclusively in either the train pool or the test pool (`StratifiedGroupKFold`).
   - **Malignant Preservation**: ALL 393 malignant images are retained across the splits (enriching malignant prevalence to ~0.39% to stabilize gradient dynamics).
   - **Enriched Prevalence**: Because prevalence is enriched relative to the 0.098% full-dataset base rate, ranking metrics (pAUC and ROC-AUC) must be used rather than unadjusted raw accuracy.