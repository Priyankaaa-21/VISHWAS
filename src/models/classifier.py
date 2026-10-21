"""
Vishwas Model Architectures: classifier.py

Implements:
1. Image-only classification: Pretrained EfficientNet-B0 or ResNet50 with custom diagnostic head.
2. Multimodal Fusion: Concatenates visual embeddings with an MLP embedding over [ITA value, skin-tone one-hot].
3. Gated Fusion: Learns a dynamic gating mechanism balancing visual features and objective skin-tone typology.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torchvision.models as tv_models

logger = logging.getLogger("vishwas.models")


class ImageOnlyClassifier(nn.Module):
    """Image-only neural network classifier with configurable vision backbone."""

    def __init__(
        self,
        num_classes: int,
        backbone_name: str = "efficientnet_b0",
        pretrained: bool = True,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.backbone_name = backbone_name.lower()
        self.num_classes = num_classes

        # Load backbone with torchvision for robust offline availability
        if "efficientnet" in self.backbone_name:
            weights = tv_models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            self.backbone = tv_models.efficientnet_b0(weights=weights)
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
        elif "resnet50" in self.backbone_name:
            weights = tv_models.ResNet50_Weights.DEFAULT if pretrained else None
            self.backbone = tv_models.resnet50(weights=weights)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        else:
            weights = tv_models.ResNet18_Weights.DEFAULT if pretrained else None
            self.backbone = tv_models.resnet18(weights=weights)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()

        self.embed_dim = in_features
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts dense feature vector before the final classification head."""
        return self.backbone(x)

    def forward(
        self,
        image: torch.Tensor,
        group_one_hot: Optional[torch.Tensor] = None,
        ita_val: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        features = self.extract_features(image)
        logits = self.classifier(features)
        return logits


class SkinToneFusionClassifier(nn.Module):
    """
    Multimodal fusion classifier that combines deep visual representations
    with objective skin-tone features (Individual Typology Angle + one-hot group).
    """

    def __init__(
        self,
        num_classes: int,
        backbone_name: str = "efficientnet_b0",
        pretrained: bool = True,
        dropout: float = 0.2,
        tabular_dim: int = 4,  # 3 one-hot + 1 continuous ITA
        tabular_embed_dim: int = 32,
    ) -> None:
        super().__init__()
        self.base_model = ImageOnlyClassifier(
            num_classes=num_classes,
            backbone_name=backbone_name,
            pretrained=pretrained,
            dropout=dropout,
        )
        self.embed_dim = self.base_model.embed_dim

        # Small MLP over [ita_val, group_one_hot]
        self.tabular_mlp = nn.Sequential(
            nn.Linear(tabular_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Linear(32, tabular_embed_dim),
            nn.ReLU(inplace=True),
        )

        fused_dim = self.embed_dim + tabular_embed_dim
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(fused_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout / 2.0),
            nn.Linear(128, num_classes),
        )

    def forward(
        self,
        image: torch.Tensor,
        group_one_hot: torch.Tensor,
        ita_val: torch.Tensor,
    ) -> torch.Tensor:
        img_feats = self.base_model.extract_features(image)

        # Reshape ita_val if needed: (B,) -> (B, 1)
        if ita_val.dim() == 1:
            ita_val = ita_val.unsqueeze(1)

        tab_input = torch.cat([group_one_hot, ita_val], dim=1)
        tab_feats = self.tabular_mlp(tab_input)

        # Concatenate visual and tabular embeddings
        fused = torch.cat([img_feats, tab_feats], dim=1)
        logits = self.classifier(fused)
        return logits


class GatedFusionClassifier(nn.Module):
    """
    Gated Multimodal Fusion classifier that dynamically weights visual features
    and objective skin tone metrics via a sigmoid attention gate.
    """

    def __init__(
        self,
        num_classes: int,
        backbone_name: str = "efficientnet_b0",
        pretrained: bool = True,
        dropout: float = 0.2,
        tabular_dim: int = 4,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.base_model = ImageOnlyClassifier(
            num_classes=num_classes,
            backbone_name=backbone_name,
            pretrained=pretrained,
            dropout=dropout,
        )
        self.embed_dim = self.base_model.embed_dim

        self.tab_encoder = nn.Sequential(
            nn.Linear(tabular_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, self.embed_dim),
        )

        # Sigmoid gate over concatenated representation
        self.gate = nn.Sequential(
            nn.Linear(self.embed_dim * 2, self.embed_dim),
            nn.Sigmoid(),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(self.embed_dim, num_classes),
        )

    def forward(
        self,
        image: torch.Tensor,
        group_one_hot: torch.Tensor,
        ita_val: torch.Tensor,
    ) -> torch.Tensor:
        img_feats = self.base_model.extract_features(image)

        if ita_val.dim() == 1:
            ita_val = ita_val.unsqueeze(1)

        tab_input = torch.cat([group_one_hot, ita_val], dim=1)
        tab_projected = self.tab_encoder(tab_input)

        # Compute dynamic gating weights
        combined = torch.cat([img_feats, tab_projected], dim=1)
        g = self.gate(combined)

        # Gated combination
        fused = g * img_feats + (1.0 - g) * tab_projected
        logits = self.classifier(fused)
        return logits


try:
    from vishwas.src.models.milk10k_classifier import build_milk10k_model as _build_milk10k_model_vishwas
except ImportError:
    from src.models.milk10k_classifier import build_milk10k_model as _build_milk10k_model_vishwas


def build_model(
    model_type: str = "image_only",
    backbone: str = "efficientnet_b0",
    num_classes: int = 9,
    pretrained: bool = True,
    dropout: float = 0.2,
    fusion_type: str = "concat",
    tabular_dim: int = 13,
) -> nn.Module:
    """Factory function for building Vishwas models."""
    m_type = model_type.lower().strip()
    milk10k_types = {
        "dermoscopic_only", "dermoscopic", "dermo_only",
        "clinical_only", "clinical", "clin_only",
        "two_image_fusion", "image_fusion", "dual_image",
        "two_image_tabular_fusion", "tabular_fusion", "full_fusion", "multimodal",
    }
    if m_type in milk10k_types:
        return _build_milk10k_model_vishwas(
            model_name=m_type,
            num_classes=num_classes,
            backbone=backbone,
            pretrained=pretrained,
            dropout=dropout,
            fusion_type=fusion_type,
            tabular_dim=tabular_dim,
        )
    elif m_type == "fusion":
        return SkinToneFusionClassifier(
            num_classes=num_classes,
            backbone_name=backbone,
            pretrained=pretrained,
            dropout=dropout,
        )
    elif m_type == "gated_fusion":
        return GatedFusionClassifier(
            num_classes=num_classes,
            backbone_name=backbone,
            pretrained=pretrained,
            dropout=dropout,
        )
    else:
        return ImageOnlyClassifier(
            num_classes=num_classes,
            backbone_name=backbone,
            pretrained=pretrained,
            dropout=dropout,
        )



if __name__ == "__main__":
    dummy_img = torch.randn(2, 3, 224, 224)
    dummy_grp = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    dummy_ita = torch.tensor([0.45, 0.25])

    for m_type in ["image_only", "fusion", "gated_fusion"]:
        model = build_model(model_type=m_type, backbone="efficientnet_b0", num_classes=9)
        out = model(dummy_img, dummy_grp, dummy_ita)
        print(f"Model '{m_type}' output logits shape: {out.shape}")
