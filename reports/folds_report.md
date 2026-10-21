# MILK10k 5-Fold Cross-Validation Partition Report

> **Date**: 2026-10-20 | **Stratification**: Diagnosis + Tone Group (Fallback: Diagnosis-only for cells < 10) | **Seed**: 42

This report documents the lesion-level 5-fold cross-validation partition constructed for the **MILK10k** dataset, ensuring leakage-free model evaluation, robust representation across all 11 diagnostic classes, and subgroup balance across skin tone groups.

## 1. Executive Summary & Partition Integrity

| Partition Property | Configured / Target | Audited Value | Status |
| :--- | :--- | :--- | :---: |
| **Total Lesions Partitioned** | 5,240 | 5,240 | PASS |
| **Total Images Accounted For (2x)** | 10,480 | 10,480 (10,480 in metadata) | PASS |
| **Number of Folds** | 5 | 5 | PASS |
| **Grouping Key** | `lesion_id` | `lesion_id` (100% paired) | PASS |
| **Fold Size Variation Range** | ≤ 5.0% | **0.00%** (1048 - 1048 lesions) | PASS |
| **Lesion Overlap Across Folds** | 0 | 0 | PASS |
| **Tone Group Mapping Scheme** | Configured `tone_groups` | `{0,1,2}->G1, {3}->G2, {4,5}->G3` | PASS |

> [!IMPORTANT]
> **Patient-Level Leakage Caveat**: Note that the MILK10k dataset does not provide a `patient_id` field. While grouping by `lesion_id` guarantees that both images of any lesion are strictly contained within the same fold, patient-level leakage across multiple distinct lesions from the same patient cannot be definitively excluded.

## 2. Tone Group Mapping & Distribution

In compliance with guidelines, groups are designated objectively as **G1, G2, G3**, and **unknown** without subjective light/medium/dark descriptors:

| Tone Group | Assigned `skin_tone_class` Values | Lesion Count | Percentage | Image Count (2x) |
| :--- | :--- | :--- | :--- | :--- |
| **`G1`** | `{0, 1, 2}` | 617 | 11.77% | 1,234 |
| **`G2`** | `{3}` | 3,174 | 60.57% | 6,348 |
| **`G3`** | `{4, 5}` | 1,449 | 27.65% | 2,898 |

## 3. Lesion Counts Per Fold

| Fold Index | Lesions Count | Percentage of Dataset | Image Count (2x) | Difference from Mean |
| :---: | :--- | :--- | :--- | :--- |
| **Fold 0** | 1,048 | 20.00% | 2,096 | +0.0 lesions (0.00%) |
| **Fold 1** | 1,048 | 20.00% | 2,096 | +0.0 lesions (0.00%) |
| **Fold 2** | 1,048 | 20.00% | 2,096 | +0.0 lesions (0.00%) |
| **Fold 3** | 1,048 | 20.00% | 2,096 | +0.0 lesions (0.00%) |
| **Fold 4** | 1,048 | 20.00% | 2,096 | +0.0 lesions (0.00%) |

## 4. Tone Group Counts Per Fold

| Tone Group | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 | **Total** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`G1`** | 115 | 135 | 131 | 135 | 101 | **617** |
| **`G2`** | 655 | 622 | 633 | 616 | 648 | **3,174** |
| **`G3`** | 278 | 291 | 284 | 297 | 299 | **1,449** |
| **Total** | 1,048 | 1,048 | 1,048 | 1,048 | 1,048 | **5,240** |

## 5. Diagnosis Class Counts Per Fold

| Diagnosis Class | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 | **Total** | Proportion |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`AKIEC`** | 69 | 49 | 66 | 57 | 62 | **303** | 5.78% |
| **`BCC`** | 505 | 480 | 509 | 490 | 538 | **2,522** | 48.13% |
| **`BEN_OTH`** | 8 | 12 | 10 | 7 | 7 | **44** | 0.84% |
| **`BKL`** | 104 | 117 | 108 | 120 | 95 | **544** | 10.38% |
| **`DF`** | 9 | 11 | 10 | 11 | 11 | **52** | 0.99% |
| **`INF`** | 16 | 11 | 4 | 10 | 9 | **50** | 0.95% |
| **`MAL_OTH`** | 2 | 2 | 1 | 2 | 2 | **9** | 0.17% |
| **`MEL`** | 85 | 101 | 86 | 95 | 83 | **450** | 8.59% |
| **`NV`** | 144 | 159 | 159 | 144 | 140 | **746** | 14.24% |
| **`SCCKA`** | 100 | 96 | 86 | 97 | 94 | **473** | 9.03% |
| **`VASC`** | 6 | 10 | 9 | 15 | 7 | **47** | 0.90% |
| **Total** | 1,048 | 1,048 | 1,048 | 1,048 | 1,048 | **5,240** | 100.00% |

## 6. Subgroup Extremes & Sparsity Audit Per Fold

We evaluated the smallest (diagnosis x tone_group) cells in each held-out fold, and flagged any diagnostic class having fewer than 2 lesions in a given fold:

| Fold Index | Smallest Cell Count (Overall) | Smallest Non-Zero Cell | Cell Details (Diagnosis x Tone Group) | Sparsity Flag (Class < 2 Lesions) |
| :---: | :---: | :---: | :--- | :--- |
| **Fold 0** | 0 | 1 | `BEN_OTH x G3` (1), `DF x G1` (1), `MAL_OTH x G1` (1) (+2 more) | None (All classes ≥ 2) |
| **Fold 1** | 0 | 1 | `SCCKA x G1` (1) | None (All classes ≥ 2) |
| **Fold 2** | 0 | 1 | `INF x G1` (1), `INF x G2` (1), `MAL_OTH x G1` (1) (+1 more) | **FLAG: `MAL_OTH` (1)** |
| **Fold 3** | 0 | 2 | `BEN_OTH x G3` (2), `DF x G2` (2), `INF x G3` (2) (+1 more) | None (All classes ≥ 2) |
| **Fold 4** | 0 | 1 | `DF x G1` (1), `SCCKA x G1` (1), `VASC x G1` (1) (+1 more) | None (All classes ≥ 2) |

> [!WARNING]
> **Sparse Subgroup Flags Encountered**:
> - In **Fold 2**, diagnosis **`MAL_OTH`** has only **1 lesion** (< 2 lesions).
> *Explanation*: `MAL_OTH` has only 9 lesions total in the entire dataset. Dividing 9 lesions across 5 folds yields [2, 2, 1, 2, 2]. Hence, exactly one fold receives 1 lesion. All remaining 10 diagnoses have ≥ 4 lesions in every fold.

## 7. Cross-Validation & Evaluation Protocol for Steps 3 and 4

To ensure zero test contamination while producing high-power, calibration-grounded fairness audits:
1. **Out-of-Fold (OOF) Training Loop**:
   - For each fold $k \in \{0, 1, 2, 3, 4\}$:
     - Use Fold $k$ as the held-out test fold.
     - Take the remaining 4 training folds (4,192 lesions) and extract an internal validation slice (grouped by lesion) for early stopping on macro-F1.
     - Train model with selected bias mitigation (class-weighted, balanced sampler, or group-weighted).
     - Predict on Fold $k$ and log outputs.
2. **OOF Prediction Pooling**:
   - Pool all 5 held-out fold predictions into a single, unified `reports/<run>/predictions.csv` covering all 5,240 lesions (10,480 images).
3. **Fairness Metrics & Reliability Curves**:
   - Evaluate Macro-F1, ROC-AUC, per-class recall, and Expected Calibration Error (ECE) across pooled predictions.
   - Group evaluation by `tone_group` (G1, G2, G3) with sample size $n$ documented for each group.
   - Compute 95% bootstrap confidence intervals (1,000 resamples).
   - Emit a formal warning for any `tone_group x class` subgroup containing fewer than 10 lesions.
