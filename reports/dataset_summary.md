# Vishwas Project Dataset Summary & Architectural Integration

This document summarizes the datasets utilized across the two-stage training, fairness auditing, and clinical diagnostic workflow in the **Vishwas** dermatological AI framework.

---

## 1. Dataset Breakdown & Usage Matrix

| Dataset Component | Total Images in Project | Sample Type / Resolution | Diagnostic Classes / Task | Patient Partitioning | Demographic / Skin-Tone Annotations | Project Stage & Purpose |
| :--- | :---: | :--- | :--- | :--- | :--- | :--- |
| **SLICE-3D (ISIC 2024 Subset)** | **`100,000`** | 3D-TBP lesion crops ($128 \times 128$) | Binary classification (Benign vs. Malignant) | **Strictly grouped by patient** (no patient shared across train/val/test) | Age, Sex, Anatomical Site (**No skin-tone labels**) | **Stage-1 Pretraining**: Domain-specific dermatological feature learning |
| **MILK10k Dataset** | **`10,480`** | Paired clinical close-up + dermoscopic ($128 \times 128$) | 11 diagnostic classes (10 headline + 1 rare) | **5-Fold Cross-Validation** (lesion & patient paired) | Confirmed `skin_tone_class` (0 to 5) $\implies$ Groups $G_1, G_2, G_3$ | **Stage-2 Fine-Tuning & Primary Fairness Audit** |
| **Combined Project Total** | **`110,480`** | Multimodal clinical + dermoscopic + 3D-TBP crops | Binary detection $\to$ 11-class differential diagnosis | Disjoint patient cohorts across folds/splits | Comprehensive clinical & demographic variables | **Complete Two-Stage Multimodal AI Pipeline** |

---

## 2. Stage-1 Dataset: Curated SLICE-3D 100,000-Image Subset

* **Source Cohort**: Drawn from the full **401,059** lesion crops of the **ISIC 2024 Challenge (SLICE-3D 3D-TBP)** dataset.
* **Fixed Architecture Split**:
  * **Train Set**: `67,500` images (267 malignant, 67,233 benign, 702 unique patients)
  * **Validation Set (10% of train pool)**: `7,500` images (29 malignant, 7,471 benign, 76 unique patients)
  * **Subtotal Train Pool**: `75,000` images (296 malignant, 74,704 benign, 778 unique patients)
  * **Test Set**: `25,000` images (97 malignant, 24,903 benign, 258 unique patients)
* **Zero Patient Overlap**: Stratified patient-grouped partitioning (`StratifiedGroupKFold`) ensures absolute disjointness across training, validation, and test sets ($0$ shared patients).
* **Malignant Preservation**: Retains **all 393 malignant images** present in the entire 401,059-image repository, enriching malignant prevalence to `0.3930%` (compared to the `0.0980%` baseline raw frequency).
* **Evaluation Standards**: Because the dataset is prevalence-enriched, performance is audited via ranking metrics (**pAUC above 80% TPR** and **ROC-AUC**) with 1,000 patient-clustered bootstrap resamples. Diagnostic accuracy is excluded.

---

## 3. Stage-2 Dataset: MILK10k Multimodal Benchmark

* **Cohort Composition**: `10,480` images representing `5,240` unique lesions.
* **Dual-View Modalities**: Each lesion has exactly one **clinical close-up** image and one **dermoscopic** image.
* **Structured Clinical Metadata**: 13-dimensional standardized tabular tensor comprising patient approximate age, sex, and anatomical site categories.
* **Evaluation Framework**: Certified 5-fold cross-validation with zero lesion leakage between training, internal validation, and held-out test folds.
* **Diagnostic Spectrum**: 11 clinical disease categories (`AKIEC`, `BCC`, `BEN_OTH`, `BKL`, `DF`, `INF`, `MAL_OTH`, `MEL`, `NV`, `SCCKA`, `VASC`).

---

## 4. Methodological Note on Demographic Fairness Evaluation

> [!IMPORTANT]
> **Fairness Auditing Scope**:
>
> In strict accordance with rigorous clinical auditing standards, **subgroup fairness and demographic disparity auditing is performed exclusively on the MILK10k benchmark**.
>
> **Reasoning**: The SLICE-3D (ISIC 2024) dataset contains demographic fields for patient age, sex, and anatomical site, but **contains zero skin-tone labels** (neither Fitzpatrick phototype nor Individual Typology Angle spectrophotometric values). Therefore, tone-group stratification ($G_1 = \{0,1,2\}$, $G_2 = \{3\}$, $G_3 = \{4,5\}$) is conducted exclusively on MILK10k, where confirmed ground truth `skin_tone_class` annotations exist for all lesions.
