"""
VISHWAS Explainability Module: src/explain/gradcam.py

Provides:
1. Grad-CAM and Grad-CAM++ implementation on the last convolutional layer (conv_head).
2. Works seamlessly across ClinicalOnlyClassifier, TwoImageTabularFusionClassifier,
   and standalone feature extractors.
3. Heatmap generation and RGBA alpha-overlay blending for top-3 candidate classes.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import matplotlib.cm as cm
except ImportError:
    cm = None


class GradCAM:
    """
    Grad-CAM & Grad-CAM++ extractor targeting the last convolutional block of visual backbones.
    """

    def __init__(
        self,
        model: nn.Module,
        target_layer: Optional[nn.Module] = None,
        branch: str = "clinical",  # "clinical" | "dermoscopic"
    ) -> None:
        self.model = model
        self.branch = branch
        self.target_layer = target_layer or self._find_target_conv_layer(model, branch)

        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None

        self._register_hooks()

    def _find_target_conv_layer(self, model: nn.Module, branch: str) -> nn.Module:
        """Finds last convolutional layer in encoder."""
        encoder = None
        if hasattr(model, "encoder"):
            encoder = model.encoder
        elif branch == "clinical" and hasattr(model, "clinical_encoder"):
            encoder = model.clinical_encoder
        elif branch == "dermoscopic" and hasattr(model, "dermoscopic_encoder"):
            encoder = model.dermoscopic_encoder

        if encoder is not None:
            if hasattr(encoder, "conv_head"):
                return encoder.conv_head
            elif hasattr(encoder, "blocks"):
                return encoder.blocks[-1]
            elif hasattr(encoder, "features"):
                return encoder.features[-1]

        # Fallback: traverse modules in reverse order to find last Conv2d
        for mod in reversed(list(model.modules())):
            if isinstance(mod, nn.Conv2d):
                return mod

        raise ValueError("Could not automatically locate a target Conv2d layer.")

    def _register_hooks(self) -> None:
        def forward_hook(module: nn.Module, input: Any, output: torch.Tensor) -> None:
            self.activations = output.detach()

        def backward_hook(module: nn.Module, grad_input: Any, grad_output: Tuple[torch.Tensor, ...]) -> None:
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate_cam(
        self,
        clin_tensor: Optional[torch.Tensor] = None,
        dermo_tensor: Optional[torch.Tensor] = None,
        tab_tensor: Optional[torch.Tensor] = None,
        target_class_idx: int = 0,
        use_gradcam_plus_plus: bool = False,
    ) -> np.ndarray:
        """
        Computes Grad-CAM or Grad-CAM++ heatmap for specified target class index.
        Returns 2D float32 array in [0.0, 1.0].
        """
        self.model.eval()
        self.model.zero_grad()

        # Determine forward call arguments based on model signature
        dev = next(self.model.parameters()).device
        if clin_tensor is not None:
            clin_tensor = clin_tensor.to(dev).requires_grad_(True)
        if dermo_tensor is not None:
            dermo_tensor = dermo_tensor.to(dev).requires_grad_(True)
        if tab_tensor is not None:
            tab_tensor = tab_tensor.to(dev)

        if hasattr(self.model, "clinical_encoder") and hasattr(self.model, "tabular_mlp"):
            # TwoImageTabularFusionClassifier
            c_in = clin_tensor if clin_tensor is not None else torch.zeros(1, 3, 128, 128, device=dev)
            d_in = dermo_tensor if dermo_tensor is not None else torch.zeros(1, 3, 128, 128, device=dev)
            t_in = tab_tensor if tab_tensor is not None else torch.zeros(1, 13, device=dev)
            logits = self.model(clinical_img=c_in, dermoscopic_img=d_in, tabular=t_in)
            ref_tensor = clin_tensor if self.branch == "clinical" else d_in
        elif hasattr(self.model, "clinical_encoder"):
            # TwoImageFusionClassifier
            c_in = clin_tensor if clin_tensor is not None else torch.zeros(1, 3, 128, 128, device=dev)
            d_in = dermo_tensor if dermo_tensor is not None else torch.zeros(1, 3, 128, 128, device=dev)
            logits = self.model(clinical_img=c_in, dermoscopic_img=d_in)
            ref_tensor = clin_tensor if self.branch == "clinical" else d_in
        else:
            # Single-branch classifier
            ref_tensor = clin_tensor if clin_tensor is not None else dermo_tensor
            if ref_tensor is None:
                raise ValueError("GradCAM requires at least one input tensor.")
            logits = self.model(ref_tensor)

        target_score = logits[0, target_class_idx]
        target_score.backward(retain_graph=True)

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Failed to capture activations or gradients.")

        activations = self.activations[0]  # [C, H, W]
        gradients = self.gradients[0]      # [C, H, W]

        if use_gradcam_plus_plus:
            # Grad-CAM++ formulation: higher-order gradient weighting
            grad_2 = gradients.pow(2)
            grad_3 = gradients.pow(3)
            sum_acts = activations.sum(dim=(1, 2), keepdim=True)
            eps = 1e-8

            denom = 2.0 * grad_2 + sum_acts * grad_3
            denom = torch.where(denom != 0.0, denom, torch.ones_like(denom) * eps)
            aij = grad_2 / denom

            weights = (aij * F.relu(gradients)).sum(dim=(1, 2))
        else:
            # Standard Grad-CAM: Global average pooling over gradients
            weights = gradients.mean(dim=(1, 2))  # [C]

        # Linear combination of feature maps weighted by alpha
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32, device=activations.device)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = F.relu(cam)

        # Upsample to input tensor resolution
        in_h, in_w = ref_tensor.shape[2], ref_tensor.shape[3]
        cam = cam.unsqueeze(0).unsqueeze(0)
        cam = F.interpolate(cam, size=(in_h, in_w), mode="bilinear", align_corners=False)
        cam_np = cam.squeeze().cpu().detach().numpy()

        # Min-max normalization
        min_v, max_v = cam_np.min(), cam_np.max()
        if max_v - min_v > 1e-8:
            cam_np = (cam_np - min_v) / (max_v - min_v)
        else:
            cam_np = np.zeros_like(cam_np)

        return cam_np.astype(np.float32)


def overlay_cam_on_image(
    image: Image.Image,
    cam_map: np.ndarray,
    alpha: float = 0.45,
    colormap_name: str = "jet",
) -> Image.Image:
    """
    Overlays a 2D float32 CAM map in [0, 1] onto a PIL Image with specified opacity.
    """
    w, h = image.size
    # Resize cam_map to exact image dimensions
    cam_pil = Image.fromarray((cam_map * 255.0).astype(np.uint8)).resize((w, h), Image.Resampling.BILINEAR)
    cam_resized = np.array(cam_pil, dtype=np.float32) / 255.0

    # Color map
    if cm is not None and hasattr(cm, colormap_name):
        cmap = getattr(cm, colormap_name)
        colored = (cmap(cam_resized)[:, :, :3] * 255.0).astype(np.uint8)
    else:
        # Fallback simple jet-like gradient
        colored = np.zeros((h, w, 3), dtype=np.uint8)
        colored[:, :, 0] = np.clip(cam_resized * 2.0 * 255.0, 0, 255).astype(np.uint8)
        colored[:, :, 2] = np.clip((1.0 - cam_resized) * 2.0 * 255.0, 0, 255).astype(np.uint8)

    orig_rgb = np.array(image.convert("RGB"), dtype=np.float32)
    blended = (1.0 - alpha) * orig_rgb + alpha * colored.astype(np.float32)
    blended = np.clip(blended, 0, 255).astype(np.uint8)

    return Image.fromarray(blended)
