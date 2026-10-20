# Vishwas Evaluation & Calibration Report: `dermamnist_baseline`

> [!NOTE]
> **EVALUATION MODE: Mode (b) - Overall Metrics Without Demographic Groups**
> **WHY**: This evaluation ran in Mode (b) because the DermaMNIST dataset (HAM10000 derived) does NOT contain patient-level Fitzpatrick Skin Phototype (FST) or individual typology angle (ITA) metadata.
> Subgroup demographic disparities and gap metrics cannot be evaluated without verified demographic labels.

## Overall Performance & Calibration

- **Overall Test Accuracy**: `86.80%` (95% CI: [84.9%, 88.7%])
- **Overall Macro-F1**: `0.6373` (95% CI: [0.5612, 0.6931])
- **Macro-AUC (OVR)**: `0.9652`
- **Expected Calibration Error (ECE)**: `0.0111`

## Visual Calibration Artifacts

- [Reliability Diagram (Calibration Curve)](reliability_curves.png)
- [Confusion Matrix](confusion_matrix.png)