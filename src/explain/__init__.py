"""
VISHWAS Explainability Package (Phase C)
"""

from src.explain.features import (
    compute_asymmetry,
    compute_border_irregularity,
    compute_feature_agreement,
    extract_dominant_colors,
    extract_lesion_features,
    segment_lesion,
)
from src.explain.gradcam import GradCAM, overlay_cam_on_image
from src.explain.sanity_checks import (
    generate_explainability_sanity_report,
    load_explain_sanity_results,
    run_model_randomization_test,
)

__all__ = [
    "segment_lesion",
    "extract_lesion_features",
    "compute_feature_agreement",
    "compute_asymmetry",
    "compute_border_irregularity",
    "extract_dominant_colors",
    "GradCAM",
    "overlay_cam_on_image",
    "run_model_randomization_test",
    "load_explain_sanity_results",
    "generate_explainability_sanity_report",
]
