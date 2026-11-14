# SLICE-3D Curated 100,000-Image Split Specification Report

> **Generated**: 2026-11-15 02:58:30 | **Runtime**: 1.40s | **Target**: Stage-1 Pretraining

## 1. Split Partitioning Summary

The 100,000-image dataset was partitioned using patient-grouped stratified sampling to prevent data leakage and maximize the clinical validity of pretraining and evaluation.

| Split Partition | Image Count | Unique Patients | Malignant Count | Benign Count | Malignant Prevalence (%) | Imbalance Ratio |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Train** | `67,500` | `700` | `258` | `67,242` | `0.3822%` | `260.6 : 1` |
| **Validation** | `7,500` | `100` | `30` | `7,470` | `0.4000%` | `249.0 : 1` |
| **Subtotal (Train+Val)** | **`75,000`** | **`778`** | **`288`** | **`74,712`** | **`0.3840%`** | **`259.4 : 1`** |
| **Test** | **`25,000`** | **`256`** | **`105`** | **`24,895`** | **`0.4200%`** | **`237.1 : 1`** |
| **Combined Total** | **`100,000`** | **`1,034`** | **`393`** | **`99,607`** | **`0.3930%`** | **`253.5 : 1`** |

## 2. Integrity and Disjointness Verification

| Verification Check | Target Requirement | Measured Status | Status |
| :--- | :--- | :--- | :---: |
| **Patient Separation (Train ∩ Val)** | Zero patient overlap | `0` shared patients | **PASSED** |
| **Patient Separation (Train ∩ Test)** | Zero patient overlap | `0` shared patients | **PASSED** |
| **Patient Separation (Val ∩ Test)** | Zero patient overlap | `0` shared patients | **PASSED** |
| **Total Train Pool Size** | Exactly 75,000 images | `75,000` images | **PASSED** |
| **Total Test Split Size** | Exactly 25,000 images | `25,000` images | **PASSED** |
| **Validation Ratio** | Exactly 10% of train pool (7,500) | `7,500` images | **PASSED** |
| **Malignant Case Preservation** | Retain all 393 malignant images | `393 / 393` retained | **PASSED** |

## 3. Prevalence Enrichment & Metric Selection Note

> **Important Clinical & Statistical Notice**:
>
> The baseline malignant prevalence in the raw 401,059-image SLICE-3D dataset is approximately **0.0980%** (393 malignant cases). In our 100,000-image subset, keeping all 393 malignant images while subsampling benign lesions enriches malignant prevalence to **0.3930%** (approximately 4x the raw frequency).
>
> Because prevalence is artificially enriched relative to the real-world base rate, **evaluation metrics must be prevalence-robust**: ranking-based and sensitivity-calibrated metrics (**partial AUC above 80% TPR** and **ROC-AUC**) are required. Raw diagnostic accuracy is uninformative and must not be used as an evaluation metric.