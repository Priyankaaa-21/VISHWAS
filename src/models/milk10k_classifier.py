"""
Vishwas MILK10k Model Architectures: milk10k_classifier.py

Implements multimodal dermatological classifiers for the MILK10k dataset:
(a) dermoscopic_only: Pretrained EfficientNet-B0 (or ResNet50) on dermoscopic image
(b) clinical_only: Pretrained EfficientNet-B0 (or ResNet50) on clinical close-up image
(c) two_image_fusion: Separate visual encoders for clinical and dermoscopic views, concatenation or gated fusion
(d) two_image_tabular_fusion: Separate visual encoders + tabular MLP (age, sex, site, optional MONET features)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import timm
import torch
import torch.nn as nn

logger = logging.getLogger("vishwas.models.milk10k")


def load_stage1_encoder_weights(
    model: nn.Module,
    checkpoint_path: Union[str, Path],
    init_branches: str = "clinical",
) -> List[str]:
    """
    Loads Stage-1 pretrained visual encoder weights into the specified branches.
    init_branches:
        'clinical' (default): initializes clinical_encoder only
        'both': initializes both clinical_encoder and dermoscopic_encoder
    """
    ckpt_p = Path(checkpoint_path)
    if not ckpt_p.is_absolute():
        script_dir = Path(__file__).resolve().parent
        vishwas_root = script_dir.parent.parent
        workspace_root = vishwas_root.parent
        for cand in [ckpt_p, vishwas_root / ckpt_p, workspace_root / ckpt_p, Path.cwd() / ckpt_p]:
            if cand.exists():
                ckpt_p = cand.resolve()
                break

    if not ckpt_p.exists():
        logger.warning(f"Stage-1 encoder checkpoint not found at: {ckpt_p}")
        return []

    state = torch.load(ckpt_p, map_location="cpu")
    if isinstance(state, dict) and "model_state" in state:
        state = state["model_state"]

    cleaned_state = {}
    for k, v in state.items():
        k_clean = k
        if k_clean.startswith("encoder."):
            k_clean = k_clean[len("encoder."):]
        cleaned_state[k_clean] = v

    branches = str(init_branches).lower().strip()
    loaded_branches: List[str] = []

    if branches in ["clinical", "both"]:
        if hasattr(model, "clinical_encoder"):
            model.clinical_encoder.load_state_dict(cleaned_state, strict=False)
            loaded_branches.append("clinical_encoder")

    if branches in ["both", "dermoscopic", "dermo"]:
        if hasattr(model, "dermoscopic_encoder"):
            model.dermoscopic_encoder.load_state_dict(cleaned_state, strict=False)
            loaded_branches.append("dermoscopic_encoder")

    if hasattr(model, "encoder") and not hasattr(model, "clinical_encoder"):
        model.encoder.load_state_dict(cleaned_state, strict=False)
        loaded_branches.append("encoder")

    logger.info(f"Loaded Stage-1 pretrained weights from {ckpt_p} into: {loaded_branches}")
    return loaded_branches


class DermoscopicOnlyClassifier(nn.Module):
    """Classifier operating exclusively on the dermoscopic view."""

    def __init__(
        self,
        num_classes: int = 11,
        backbone_name: str = "efficientnet_b0",
        pretrained: bool = True,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.encoder = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0)
        self.embed_dim = self.encoder.num_features
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(self.embed_dim, num_classes),
        )

    def forward(
        self,
        clinical_img: Optional[torch.Tensor] = None,
        dermoscopic_img: Optional[torch.Tensor] = None,
        tabular: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        img = dermoscopic_img if dermoscopic_img is not None else clinical_img
        if img is None:
            raise ValueError("DermoscopicOnlyClassifier requires dermoscopic_img.")
        feats = self.encoder(img)
        logits = self.classifier(feats)
        return logits


class ClinicalOnlyClassifier(nn.Module):
    """Classifier operating exclusively on the clinical close-up view."""

    def __init__(
        self,
        num_classes: int = 11,
        backbone_name: str = "efficientnet_b0",
        pretrained: bool = True,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.encoder = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0)
        self.embed_dim = self.encoder.num_features
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(self.embed_dim, num_classes),
        )

    def forward(
        self,
        clinical_img: Optional[torch.Tensor] = None,
        dermoscopic_img: Optional[torch.Tensor] = None,
        tabular: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        img = clinical_img if clinical_img is not None else dermoscopic_img
        if img is None:
            raise ValueError("ClinicalOnlyClassifier requires clinical_img.")
        feats = self.encoder(img)
        logits = self.classifier(feats)
        return logits


class TwoImageFusionClassifier(nn.Module):
    """
    Multiview classifier utilizing separate encoders for clinical close-up
    and dermoscopic images, combined via concatenation or gated fusion.
    """

    def __init__(
        self,
        num_classes: int = 11,
        backbone_name: str = "efficientnet_b0",
        pretrained: bool = True,
        dropout: float = 0.2,
        fusion_type: str = "concat",  # "concat" | "gated"
    ) -> None:
        super().__init__()
        self.fusion_type = fusion_type.lower()
        self.clinical_encoder = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0)
        self.dermoscopic_encoder = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0)

        enc_dim = self.clinical_encoder.num_features

        if self.fusion_type == "gated":
            # Gated fusion projecting clinical and dermoscopic features
            self.gate = nn.Sequential(
                nn.Linear(enc_dim * 2, enc_dim),
                nn.ReLU(),
                nn.Linear(enc_dim, enc_dim),
                nn.Sigmoid(),
            )
            fused_dim = enc_dim
        else:
            # Concatenation fusion
            fused_dim = enc_dim * 2

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(fused_dim, num_classes),
        )

    def forward(
        self,
        clinical_img: torch.Tensor,
        dermoscopic_img: torch.Tensor,
        tabular: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        c_feats = self.clinical_encoder(clinical_img)
        d_feats = self.dermoscopic_encoder(dermoscopic_img)

        if self.fusion_type == "gated":
            combined = torch.cat([c_feats, d_feats], dim=1)
            g = self.gate(combined)
            fused = g * c_feats + (1.0 - g) * d_feats
        else:
            fused = torch.cat([c_feats, d_feats], dim=1)

        logits = self.classifier(fused)
        return logits


class TwoImageTabularFusionClassifier(nn.Module):
    """
    Multimodal classifier fusing dual visual representations (clinical + dermoscopic)
    with structured patient demographic and clinical site tabular features.
    """

    def __init__(
        self,
        num_classes: int = 11,
        backbone_name: str = "efficientnet_b0",
        pretrained: bool = True,
        dropout: float = 0.2,
        tabular_dim: int = 13,
        tabular_embed_dim: int = 64,
        fusion_type: str = "concat",  # "concat" | "gated"
    ) -> None:
        super().__init__()
        self.fusion_type = fusion_type.lower()
        self.clinical_encoder = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0)
        self.dermoscopic_encoder = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0)

        enc_dim = self.clinical_encoder.num_features

        self.tabular_mlp = nn.Sequential(
            nn.Linear(tabular_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(64, tabular_embed_dim),
            nn.ReLU(),
        )

        if self.fusion_type == "gated":
            visual_dim = enc_dim * 2
            self.gate = nn.Sequential(
                nn.Linear(visual_dim + tabular_embed_dim, visual_dim),
                nn.ReLU(),
                nn.Linear(visual_dim, visual_dim),
                nn.Sigmoid(),
            )
            fused_dim = visual_dim + tabular_embed_dim
        else:
            fused_dim = enc_dim * 2 + tabular_embed_dim

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(fused_dim, num_classes),
        )

    def forward(
        self,
        clinical_img: torch.Tensor,
        dermoscopic_img: torch.Tensor,
        tabular: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        c_feats = self.clinical_encoder(clinical_img)
        d_feats = self.dermoscopic_encoder(dermoscopic_img)
        t_feats = self.tabular_mlp(tabular)

        if self.fusion_type == "gated":
            v_combined = torch.cat([c_feats, d_feats], dim=1)
            vt_all = torch.cat([v_combined, t_feats], dim=1)
            g = self.gate(vt_all)
            gated_visual = g * v_combined
            fused = torch.cat([gated_visual, t_feats], dim=1)
        else:
            fused = torch.cat([c_feats, d_feats, t_feats], dim=1)

        logits = self.classifier(fused)
        return logits


def build_milk10k_model(
    model_name: str,
    num_classes: int = 11,
    backbone: str = "efficientnet_b0",
    pretrained: bool = True,
    dropout: float = 0.2,
    fusion_type: str = "concat",
    tabular_dim: int = 13,
    init_encoder: Optional[Union[str, Path]] = None,
    init_branches: str = "clinical",
) -> nn.Module:
    """Factory function for creating MILK10k classifiers."""
    m_name = model_name.lower().strip()
    if m_name in ["dermoscopic_only", "dermoscopic", "dermo_only"]:
        model = DermoscopicOnlyClassifier(
            num_classes=num_classes,
            backbone_name=backbone,
            pretrained=pretrained,
            dropout=dropout,
        )
    elif m_name in ["clinical_only", "clinical", "clin_only"]:
        model = ClinicalOnlyClassifier(
            num_classes=num_classes,
            backbone_name=backbone,
            pretrained=pretrained,
            dropout=dropout,
        )
    elif m_name in ["two_image_fusion", "image_fusion", "dual_image"]:
        model = TwoImageFusionClassifier(
            num_classes=num_classes,
            backbone_name=backbone,
            pretrained=pretrained,
            dropout=dropout,
            fusion_type=fusion_type,
        )
    elif m_name in ["two_image_tabular_fusion", "tabular_fusion", "full_fusion", "multimodal"]:
        model = TwoImageTabularFusionClassifier(
            num_classes=num_classes,
            backbone_name=backbone,
            pretrained=pretrained,
            dropout=dropout,
            tabular_dim=tabular_dim,
            fusion_type=fusion_type,
        )
    else:
        raise ValueError(
            f"Unknown MILK10k model: '{model_name}'. Choose from "
            "['dermoscopic_only', 'clinical_only', 'two_image_fusion', 'two_image_tabular_fusion']."
        )

    # Optional Stage-1 weight loading
    if init_encoder is not None:
        load_stage1_encoder_weights(model, init_encoder, init_branches=init_branches)

    return model
