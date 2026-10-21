# MILK10k Dataset Audit & Integrity Report

> **Audit Date**: 2026-10-20 | **Audit Runtime**: 3.09 seconds | **Engine**: Vishwas Dataset Ingest Auditor

This report documents the rigorous data integrity, demographic, diagnosis distribution, image resolution, and cross-attribute balance audit of the **MILK10k** dataset, conducted in Step 1 of the dataset transition.

## 1. Executive Summary & Verification Matrix

| Verification Metric | Expected Value | Audited Value | Status |
| :--- | :--- | :--- | :--- |
| **Total Images in Metadata** | 10,480 | 10,480 | PASS |
| **Total Lesions in Metadata** | 5,240 | 5,240 | PASS |
| **Total Lesions in Ground Truth** | 5,240 | 5,240 | PASS |
| **Lesion ID Exact Match (Meta == GT)** | 100% | 100% | PASS |
| **Total Subdirectories on Disk** | 5,241 (5,240 lesions + 1 licenses) | 5,241 | PASS |
| **Missing Images on Disk** | 0 | 0 | PASS |
| **Corrupt / Unreadable Images** | 0 | 0 | PASS |
| **Lesions with != 2 Images** | 0 | 0 | PASS |
| **Lesions without (1 Clinical + 1 Dermoscopic)** | 0 | 0 | PASS |
| **Lesions with Inconsistent Metadata Pairs** | 0 | 0 | PASS |
| **Single-Label Consistency (sum=1.0)** | 5,240 | 5,240 | PASS |

## 2. Non-Image Files Discovered

Scanning the extracted dataset directories revealed the following extra non-image files:

### Training Folder (`data/milk10k/train_images/MILK10k_Training_Input`)
- `attribution.txt`
- `licenses/CC-BY-NC.txt`

### Test Folder (`data/milk10k/test_images/MILK10k_Test_Input`)
- `attribution.txt`
- `licenses/CC-BY-NC.txt`

> [!NOTE]
> The directory `MILK10k_Training_Input/licenses` contains `CC-BY-NC.txt`, and the root directory contains `attribution.txt`. Both folders are clean of any unindexed images or corrupted artifacts.

## 3. Lesion Image Pairing & Metadata Consistency

Every lesion in MILK10k is indexed by two rows in `MILK10k_Training_Metadata.csv`:
1. `clinical: close-up` image
2. `dermoscopic` image

- **Lesions with exactly 2 images**: **5,240 / 5,240 (100.0%)**
- **Exceptions (pair count != 2)**: **None (0)**
- **Exceptions (types != ['clinical: close-up', 'dermoscopic'])**: **None (0)**

### Metadata Consistency Between Paired Images
We evaluated whether any patient-level metadata differed between the clinical row and dermoscopic row for each lesion:

| Attribute | Discrepancies Found | Consistency Rate |
| :--- | :--- | :--- |
| `age_approx` | 0 | 100.00% |
| `sex` | 0 | 100.00% |
| `site` | 0 | 100.00% |
| `skin_tone_class` | 0 | 100.00% |

> [!TIP]
> Pair metadata is **100% congruent**. There is zero internal contradiction in patient demographics or lesion site between clinical and dermoscopic views.

## 4. Image Size Distribution & File Integrity

- **Total images verified**: 10,480
- **Corrupt or unreadable files**: **0 (None)**
- **Color mode distribution**: `{'RGB': 10480}` (100% RGB)

### Image Resolutions

| Dimensions (Width x Height) | Aspect Ratio | Image Count | Percentage |
| :--- | :--- | :--- | :--- |
| 600 x 450 | 4:3 (600x450) | 10,480 | 100.00% |

### File Size Distribution (Disk Footprint)
- **Mean**: 33.0 KB
- **Median**: 31.4 KB
- **Min / Max**: 11.5 KB / 99.1 KB
- **Interquartile Range (P25 - P75)**: 25.1 KB - 39.7 KB

## 5. Demographic & Clinical Attribute Distributions

All distributions below are reported at the **lesion level** (N = 5,240 lesions, each representing 2 images).

### 5.1 Skin Tone Class (`skin_tone_class`)

> [!IMPORTANT]
> As specified, no assumptions are made about the physiological meaning of the `skin_tone_class` scale. Raw audited integer categories `[0, 1, 2, 3, 4, 5]` are documented directly.

| Skin Tone Class | Lesion Count | Percentage | Image Count (2x) |
| :--- | :--- | :--- | :--- |
| **`3`** | 3,174 | 60.57% | 6,348 |
| **`4`** | 1,066 | 20.34% | 2,132 |
| **`2`** | 506 | 9.66% | 1,012 |
| **`5`** | 383 | 7.31% | 766 |
| **`1`** | 105 | 2.00% | 210 |
| **`0`** | 6 | 0.11% | 12 |

### 5.2 Sex (`sex`)

| Sex | Lesion Count | Percentage |
| :--- | :--- | :--- |
| `male` | 3,152 | 60.15% |
| `female` | 2,088 | 39.85% |

### 5.3 Anatomical Site (`site`)

| Anatomical Site | Lesion Count | Percentage |
| :--- | :--- | :--- |
| `trunk` | 1,925 | 36.74% |
| `head_neck_face` | 1,491 | 28.45% |
| `lower_extremity` | 1,023 | 19.52% |
| `upper_extremity` | 667 | 12.73% |
| `hand` | 72 | 1.37% |
| `Missing (NaN)` | 31 | 0.59% |
| `foot` | 23 | 0.44% |
| `genital` | 8 | 0.15% |

### 5.4 Patient Age (`age_approx`)

- **Valid Annotations**: 5,220 lesions (99.62%)
- **Missing Values**: **20 lesions (0.38%)**
- **Mean ± Std**: 61.36 ± 15.69 years
- **Median**: 65.0 years (IQR: 50.0 - 75.0)
- **Range**: [5, 85] years

| Age Range | Lesion Count | Valid Percentage |
| :--- | :--- | :--- |
| `<18` | 77 | 1.48% |
| `18-29` | 91 | 1.74% |
| `30-44` | 466 | 8.93% |
| `45-59` | 1,188 | 22.76% |
| `60-74` | 2,082 | 39.89% |
| `75+` | 1,316 | 25.21% |
| `Missing (NaN)` | 20 | 0.38% |

## 6. Ground Truth Diagnosis Distribution (11 Classes)

The 11 diagnostic classes in `MILK10k_Training_GroundTruth.csv` were converted from one-hot vectors using `argmax`:

| Rank | Diagnosis Code | Full / Common Medical Label | Lesion Count | Percentage | Cumulative % |
| :---: | :--- | :--- | :--- | :--- | :--- |
| 1 | **`BCC`** | Basal Cell Carcinoma | 2,522 | 48.13% | 48.13% |
| 2 | **`NV`** | Melanocytic Nevus | 746 | 14.24% | 62.37% |
| 3 | **`BKL`** | Benign Keratosis-like Lesion | 544 | 10.38% | 72.75% |
| 4 | **`SCCKA`** | Squamous Cell Carcinoma / Keratoacanthoma | 473 | 9.03% | 81.77% |
| 5 | **`MEL`** | Melanoma | 450 | 8.59% | 90.36% |
| 6 | **`AKIEC`** | Actinic Keratosis / Intraepithelial Carcinoma | 303 | 5.78% | 96.15% |
| 7 | **`DF`** | Dermatofibroma | 52 | 0.99% | 97.14% |
| 8 | **`INF`** | Inflammatory / Infectious Conditions | 50 | 0.95% | 98.09% |
| 9 | **`VASC`** | Vascular Lesion | 47 | 0.90% | 98.99% |
| 10 | **`BEN_OTH`** | Other Benign Lesions | 44 | 0.84% | 99.83% |
| 11 | **`MAL_OTH`** | Other Malignant Lesions | 9 | 0.17% | 100.00% |

> [!NOTE]
> The top 3 classes (`BCC`, `NV`, `BKL`) constitute **72.75%** of the entire dataset. The rarest classes (`MAL_OTH`, `BEN_OTH`, `VASC`, `INF`, `DF`) have between 9 and 52 lesions each, representing severe class imbalance.

## 7. Crosstab: Diagnosis x Skin Tone Class

Distribution of lesions across all 11 diagnoses and 6 raw skin tone classes `[0, 1, 2, 3, 4, 5]`:

| Diagnosis | Class 0 | Class 1 | Class 2 | Class 3 | Class 4 | Class 5 | **Total** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`AKIEC`** | 0 | 19 | 21 | 202 | 46 | 15 | 303 |
| **`BCC`** | 1 | 19 | 77 | 1,716 | 496 | 213 | 2,522 |
| **`BEN_OTH`** | 0 | 2 | 11 | 22 | 6 | 3 | 44 |
| **`BKL`** | 0 | 24 | 59 | 343 | 94 | 24 | 544 |
| **`DF`** | 0 | 4 | 9 | 25 | 9 | 5 | 52 |
| **`INF`** | 3 | 2 | 5 | 16 | 10 | 14 | 50 |
| **`MAL_OTH`** | 0 | 1 | 3 | 5 | 0 | 0 | 9 |
| **`MEL`** | 0 | 16 | 121 | 200 | 90 | 23 | 450 |
| **`NV`** | 2 | 16 | 185 | 319 | 172 | 52 | 746 |
| **`SCCKA`** | 0 | 1 | 5 | 304 | 135 | 28 | 473 |
| **`VASC`** | 0 | 1 | 10 | 22 | 8 | 6 | 47 |
| **`Total`** | 6 | 105 | 506 | 3,174 | 1,066 | 383 | 5,240 |

## 8. Low-Count Cell Warnings (< 10 Lesions)

> [!WARNING]
> There are **30 diagnosis x skin-tone cells** with fewer than 10 lesions (out of 66 possible cells, 45.5%). These pose severe sample size limitations for subgroup stratification and per-group test evaluations.

| Diagnosis | Skin Tone Class | Lesion Count | Evaluation Severity |
| :--- | :---: | :---: | :--- |
| `AKIEC` | `0` | **0** | EMPTY (0 samples) |
| `BCC` | `0` | **1** | CRITICAL (<= 2 samples) |
| `BEN_OTH` | `0` | **0** | EMPTY (0 samples) |
| `BEN_OTH` | `1` | **2** | CRITICAL (<= 2 samples) |
| `BEN_OTH` | `4` | **6** | MODERATE RISK (5-9 samples) |
| `BEN_OTH` | `5` | **3** | HIGH RISK (< 5 samples) |
| `BKL` | `0` | **0** | EMPTY (0 samples) |
| `DF` | `0` | **0** | EMPTY (0 samples) |
| `DF` | `1` | **4** | HIGH RISK (< 5 samples) |
| `DF` | `2` | **9** | MODERATE RISK (5-9 samples) |
| `DF` | `4` | **9** | MODERATE RISK (5-9 samples) |
| `DF` | `5` | **5** | MODERATE RISK (5-9 samples) |
| `INF` | `0` | **3** | HIGH RISK (< 5 samples) |
| `INF` | `1` | **2** | CRITICAL (<= 2 samples) |
| `INF` | `2` | **5** | MODERATE RISK (5-9 samples) |
| `MAL_OTH` | `0` | **0** | EMPTY (0 samples) |
| `MAL_OTH` | `1` | **1** | CRITICAL (<= 2 samples) |
| `MAL_OTH` | `2` | **3** | HIGH RISK (< 5 samples) |
| `MAL_OTH` | `3` | **5** | MODERATE RISK (5-9 samples) |
| `MAL_OTH` | `4` | **0** | EMPTY (0 samples) |
| `MAL_OTH` | `5` | **0** | EMPTY (0 samples) |
| `MEL` | `0` | **0** | EMPTY (0 samples) |
| `NV` | `0` | **2** | CRITICAL (<= 2 samples) |
| `SCCKA` | `0` | **0** | EMPTY (0 samples) |
| `SCCKA` | `1` | **1** | CRITICAL (<= 2 samples) |
| `SCCKA` | `2` | **5** | MODERATE RISK (5-9 samples) |
| `VASC` | `0` | **0** | EMPTY (0 samples) |
| `VASC` | `1` | **1** | CRITICAL (<= 2 samples) |
| `VASC` | `4` | **8** | MODERATE RISK (5-9 samples) |
| `VASC` | `5` | **6** | MODERATE RISK (5-9 samples) |

### Key Stratification Takeaways:
1. **Skin Tone Class 0** contains only **6 lesions in total** across the entire dataset (1 BCC, 3 INF, 2 NV, 0 for all other 8 classes). Standard multi-class stratification cannot split class 0 evenly without zero-shot splits.
2. **MAL_OTH** has only **9 lesions in total** across the entire dataset, with representation only in skin tone classes 1 (1 lesion), 2 (3 lesions), and 3 (5 lesions). Classes 0, 4, and 5 have 0 lesions.
3. In Step 2 (split generation), stratification must prioritize `diagnosis` first, and secondary stratification by `skin_tone_class` where cell counts allow (≥ 10 lesions), ensuring no empty splits.

## 9. MONET Visual Attribute Feature Analysis

The metadata includes 7 continuous visual attribute scores produced by the MONET foundation model:

| MONET Feature | Mean ± Std | Min | Max | Missing Count |
| :--- | :--- | :--- | :--- | :--- |
| `MONET_ulceration_crust` | 0.360 ± 0.232 | 0.005 | 0.988 | 0 |
| `MONET_hair` | 0.282 ± 0.170 | 0.013 | 0.949 | 0 |
| `MONET_vasculature_vessels` | 0.121 ± 0.184 | 0.000 | 0.959 | 0 |
| `MONET_erythema` | 0.356 ± 0.226 | 0.007 | 0.970 | 0 |
| `MONET_pigmented` | 0.313 ± 0.264 | 0.004 | 0.975 | 0 |
| `MONET_gel_water_drop_fluid_dermoscopy_liquid` | 0.244 ± 0.158 | 0.005 | 0.973 | 0 |
| `MONET_skin_markings_pen_ink_purple_pen` | 0.239 ± 0.221 | 0.000 | 0.990 | 0 |

> [!NOTE]
> In accordance with project instructions, MONET columns are **not** used as default inputs, but are preserved for an optional `use_monet` multimodal ablation in Step 3.

## 10. Audit Conclusion & Recommendations for Step 2

1. **Data Completeness**: 100% of the 10,480 images are present on disk, readable, valid 3-channel RGB, and uniformly sized at 600x450.
2. **Pair Disjointness**: All 5,240 lesions have exactly 1 clinical close-up and 1 dermoscopic image with congruent metadata.
3. **Split Strategy for Step 2**:
   - Lesion-level partitioning (70% train, 15% val, 15% test) must assign both images of each lesion to the exact same partition.
   - Use iterative stratified splitting (e.g. `multilabel_train_test_split` or hierarchical diagnosis-stratified grouping) to preserve diagnosis proportions across splits while balancing skin tone class representation.
   - Document that no `patient_id` exists in MILK10k metadata, so patient-level leakage across multiple lesions cannot be definitively excluded.
