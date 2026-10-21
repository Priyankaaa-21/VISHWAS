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
from typing import Any, Dict, Optional, Tuple

import timm
import torch
import torch.nn as nn

logger = logging.getLogger("vishwas.models.milk10k")


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
) -> nn.Module:
    """Factory function for creating MILK10k classifiers."""
    m_name = model_name.lower().strip()
    if m_name in ["dermoscopic_only", "dermoscopic", "dermo_only"]:
        return DermoscopicOnlyClassifier(
            num_classes=num_classes,
            backbone_name=backbone,
            pretrained=pretrained,
            dropout=dropout,
        )
    elif m_name in ["clinical_only", "clinical", "clin_only"]:
        return ClinicalOnlyClassifier(
            num_classes=num_classes,
            backbone_name=backbone,
            pretrained=pretrained,
            dropout=dropout,
        )
    elif m_name in ["two_image_fusion", "image_fusion", "dual_image"]:
        return TwoImageFusionClassifier(
            num_classes=num_classes,
            backbone_name=backbone,
            pretrained=pretrained,
            dropout=dropout,
            fusion_type=fusion_type,
        )
    elif m_name in ["two_image_tabular_fusion", "tabular_fusion", "full_fusion", "multimodal"]:
        return TwoImageTabularFusionClassifier(
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
