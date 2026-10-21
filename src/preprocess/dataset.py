"""
Vishwas Unified Dataset & DataLoader Module: dataset.py

Supports multiple dermatological datasets:
1. DermaMNIST (Corrected):
   - 28x28 or 224x224 spatial resolutions from .npz
   - Upscaling for 28x28 inputs (to 64px or 128px) for standard CNN backbones
   - 7 diagnostic condition classes
   - Skin-tone columns optional (returns skin_tone_group = None)
2. Fitzpatrick17k-C:
   - Clinical images with 9-partition, 3-partition, or 114 detailed condition classes
   - Skin-tone groups (Light, Medium, Dark) and continuous Individual Typology Angle (ITA)
3. Configurable augmentations:
   - 'standard': RandomResizedCrop, HorizontalFlip, VerticalFlip, Rotation, mild brightness jitter (strictly no hue jitter)
   - 'colour_preserving': strictly geometric transforms (no colour/brightness changes)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import yaml

# Standard DermaMNIST 7 disease classes
DERMAMNIST_CLASSES = [
    "actinic keratoses and intraepithelial carcinoma",
    "basal cell carcinoma",
    "benign keratosis-like lesions",
    "dermatofibroma",
    "melanoma",
    "melanocytic nevi",
    "vascular lesions",
]

DERMAMNIST_LABEL_TO_IDX = {name: i for i, name in enumerate(DERMAMNIST_CLASSES)}
DERMAMNIST_IDX_TO_LABEL = {i: name for i, name in enumerate(DERMAMNIST_CLASSES)}

GROUP_TO_IDX = {"light": 0, "medium": 1, "dark": 2, "unknown": 1}
IDX_TO_GROUP = {0: "light", 1: "medium", 2: "dark"}


class DermaMNISTDataset(Dataset):
    """PyTorch Dataset for DermaMNIST loaded from NumPy .npz arrays."""

    def __init__(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        split_name: str = "train",
        transform: Optional[transforms.Compose] = None,
    ) -> None:
        self.images = images  # (N, H, W, 3) uint8
        self.labels = labels.squeeze()  # (N,) uint8
        self.split_name = split_name
        self.transform = transform
        self.num_classes = len(DERMAMNIST_CLASSES)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        img_arr = self.images[idx]
        pil_img = Image.fromarray(img_arr)

        if self.transform:
            img_tensor = self.transform(pil_img)
        else:
            img_tensor = transforms.ToTensor()(pil_img)

        label_idx = int(self.labels[idx])
        label_name = DERMAMNIST_IDX_TO_LABEL.get(label_idx, f"class_{label_idx}")

        return {
            "image": img_tensor,
            "label": torch.tensor(label_idx, dtype=torch.long),
            "label_name": label_name,
            "image_id": f"dermamnist_{self.split_name}_{idx:05d}",
            "skin_tone_group": None,  # Optional: absent in DermaMNIST
            "group_idx": torch.tensor(-1, dtype=torch.long),
            "group_one_hot": torch.zeros(3, dtype=torch.float32),
            "ita_val": torch.tensor(0.0, dtype=torch.float32),
            "has_skin_tone": False,
            "fitzpatrick": -1,
        }


class FitzpatrickDataset(Dataset):
    """PyTorch Dataset for Fitzpatrick17k-C images."""

    def __init__(
        self,
        df: pd.DataFrame,
        image_dir: Path,
        target_col: str = "nine_partition_label",
        label_to_idx: Optional[Dict[str, int]] = None,
        transform: Optional[transforms.Compose] = None,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.target_col = target_col
        self.transform = transform

        if label_to_idx is None:
            unique_labels = sorted(self.df[target_col].dropna().unique())
            self.label_to_idx = {lbl: idx for idx, lbl in enumerate(unique_labels)}
        else:
            self.label_to_idx = label_to_idx

        self.idx_to_label = {idx: lbl for lbl, idx in self.label_to_idx.items()}
        self.num_classes = len(self.label_to_idx)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        filepath_str = str(row.get("filepath", "")).strip()
        img_path = self.image_dir / filepath_str

        # If not at filepath directly, check alternative image_path column
        if not img_path.exists() and "image_path" in row and pd.notna(row["image_path"]):
            img_path = Path(str(row["image_path"]))

        try:
            with Image.open(img_path) as pil_img:
                img = pil_img.convert("RGB")
        except Exception:
            img = Image.new("RGB", (224, 224), (128, 128, 128))

        if self.transform:
            img_tensor = self.transform(img)
        else:
            img_tensor = transforms.ToTensor()(img)

        label_str = str(row[self.target_col])
        target_idx = self.label_to_idx.get(label_str, 0)

        # Skin tone annotations
        grp_str = str(row.get("ita_group", row.get("fitzpatrick_group", "unknown"))).lower()
        has_skin_tone = grp_str in ("light", "medium", "dark")
        group_idx = GROUP_TO_IDX.get(grp_str, -1)

        group_one_hot = torch.zeros(3, dtype=torch.float32)
        if group_idx >= 0:
            group_one_hot[group_idx] = 1.0

        ita_raw = float(row.get("ita_value", 30.0))
        if np.isnan(ita_raw):
            ita_raw = 30.0

        return {
            "image": img_tensor,
            "label": torch.tensor(target_idx, dtype=torch.long),
            "label_name": label_str,
            "image_id": filepath_str or str(row.get("md5hash", f"img_{idx}")),
            "skin_tone_group": grp_str if has_skin_tone else None,
            "group_idx": torch.tensor(group_idx, dtype=torch.long),
            "group_one_hot": group_one_hot,
            "ita_val": torch.tensor(ita_raw / 100.0, dtype=torch.float32),
            "has_skin_tone": has_skin_tone,
            "fitzpatrick": int(row.get("fitzpatrick", -1)),
        }


def vishwas_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Custom collate function supporting optional None values for skin-tone fields."""
    images = torch.stack([b["image"] for b in batch])
    labels = torch.stack([b["label"] for b in batch])
    group_one_hots = torch.stack([b["group_one_hot"] for b in batch])
    ita_vals = torch.stack([b["ita_val"] for b in batch])
    group_indices = torch.stack([b["group_idx"] for b in batch])

    return {
        "image": images,
        "label": labels,
        "label_name": [b["label_name"] for b in batch],
        "image_id": [b["image_id"] for b in batch],
        "skin_tone_group": [b["skin_tone_group"] for b in batch],
        "group_idx": group_indices,
        "group_one_hot": group_one_hots,
        "ita_val": ita_vals,
        "has_skin_tone": [b["has_skin_tone"] for b in batch],
        "fitzpatrick": [b["fitzpatrick"] for b in batch],
    }


def get_transforms(
    augmentation_mode: str = "standard",
    target_size: int = 64,
    is_low_res: bool = True,
) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Constructs train and evaluation transforms.
    For 28x28 images, upscales to target_size (64 or 128) for effective CNN feature extraction.
    Mild brightness jitter only (strictly NO hue jitter to preserve color/pigmentation fidelity).
    """
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    if is_low_res:
        eval_transform = transforms.Compose([
            transforms.Resize((target_size, target_size)),
            transforms.ToTensor(),
            normalize,
        ])

        if augmentation_mode == "colour_preserving":
            train_transform = transforms.Compose([
                transforms.Resize((target_size, target_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ToTensor(),
                normalize,
            ])
        else:
            # Standard: mild brightness only
            train_transform = transforms.Compose([
                transforms.Resize((target_size, target_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.08),  # strictly no hue distortion
                transforms.ToTensor(),
                normalize,
            ])
    else:
        # Standard high-res (224x224)
        eval_transform = transforms.Compose([
            transforms.Resize(int(target_size * 1.14)),
            transforms.CenterCrop(target_size),
            transforms.ToTensor(),
            normalize,
        ])

        if augmentation_mode == "colour_preserving":
            train_transform = transforms.Compose([
                transforms.RandomResizedCrop(target_size, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ToTensor(),
                normalize,
            ])
        else:
            train_transform = transforms.Compose([
                transforms.RandomResizedCrop(target_size, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.08),
                transforms.ToTensor(),
                normalize,
            ])

    return train_transform, eval_transform


def load_dermamnist_splits(
    npz_path: Path,
    resolution: int = 28,
    upscale_size: int = 64,
    augmentation: str = "standard",
    batch_size: int = 64,
    num_workers: int = 2,
) -> Tuple[Dict[str, DataLoader], Dict[str, Any]]:
    """Loads and auto-detects DermaMNIST .npz files."""
    if not npz_path.exists():
        raise FileNotFoundError(f"DermaMNIST NPZ file not found at {npz_path}")

    data = np.load(npz_path)
    keys = list(data.keys())

    # Auto-detect keys
    tr_img_key = next((k for k in keys if "train" in k and "image" in k), "train_images")
    tr_lbl_key = next((k for k in keys if "train" in k and "label" in k), "train_labels")
    val_img_key = next((k for k in keys if "val" in k and "image" in k), "val_images")
    val_lbl_key = next((k for k in keys if "val" in k and "label" in k), "val_labels")
    test_img_key = next((k for k in keys if "test" in k and "image" in k), "test_images")
    test_lbl_key = next((k for k in keys if "test" in k and "label" in k), "test_labels")

    train_imgs, train_lbls = data[tr_img_key], data[tr_lbl_key]
    val_imgs, val_lbls = data[val_img_key], data[val_lbl_key]
    test_imgs, test_lbls = data[test_img_key], data[test_lbl_key]

    print("\n" + "=" * 70)
    print(f"        DERMAMNIST LOADER ({resolution}x{resolution} resolution)")
    print("=" * 70)
    print(f"Loaded from: {npz_path}")
    print(f"Auto-detected keys: {keys}")
    print(f"Train images: {train_imgs.shape}, labels: {train_lbls.shape}")
    print(f"Val images  : {val_imgs.shape}, labels: {val_lbls.shape}")
    print(f"Test images : {test_imgs.shape}, labels: {test_lbls.shape}")

    # Print class counts per split
    print("\n--- Class Counts Per Split (7 DermaMNIST Classes) ---")
    splits_data = [("train", train_lbls), ("val", val_lbls), ("test", test_lbls)]
    counts_table = []
    for split_name, lbls in splits_data:
        unique, counts = np.unique(lbls, return_counts=True)
        split_dict = dict(zip(unique, counts))
        for c_idx, c_name in enumerate(DERMAMNIST_CLASSES):
            counts_table.append({
                "Split": split_name,
                "Class ID": c_idx,
                "Class Name": c_name,
                "Count": split_dict.get(c_idx, 0),
            })
    counts_df = pd.DataFrame(counts_table)
    pivot_df = counts_df.pivot(index=["Class ID", "Class Name"], columns="Split", values="Count").fillna(0).astype(int)
    print(pivot_df.to_string())
    print("=" * 70 + "\n")

    is_low_res = (resolution == 28)
    target_dim = upscale_size if is_low_res else 224

    train_tf, eval_tf = get_transforms(augmentation, target_size=target_dim, is_low_res=is_low_res)

    train_ds = DermaMNISTDataset(train_imgs, train_lbls, split_name="train", transform=train_tf)
    val_ds = DermaMNISTDataset(val_imgs, val_lbls, split_name="val", transform=eval_tf)
    test_ds = DermaMNISTDataset(test_imgs, test_lbls, split_name="test", transform=eval_tf)

    dataloaders = {
        "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=False, collate_fn=vishwas_collate_fn),
        "val": DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=vishwas_collate_fn),
        "test": DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=vishwas_collate_fn),
    }

    info = {
        "dataset_name": "dermamnist",
        "has_skin_tone": False,
        "num_classes": 7,
        "classes": DERMAMNIST_CLASSES,
        "label_to_idx": DERMAMNIST_LABEL_TO_IDX,
        "idx_to_label": DERMAMNIST_IDX_TO_LABEL,
        "train_size": len(train_ds),
        "val_size": len(val_ds),
        "test_size": len(test_ds),
        "resolution": resolution,
        "upscale_size": target_dim,
        "augmentation": augmentation,
    }
    return dataloaders, info


def build_dataloaders(
    config_path: str = "vishwas/configs/config.yaml",
    image_dir_override: Optional[str] = None,
    dataset_override: Optional[str] = None,
) -> Tuple[Dict[str, DataLoader], Dict[str, Any]]:
    """Unified entry point to construct DataLoaders for DermaMNIST or Fitzpatrick17k-C."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    dataset_type = (dataset_override or cfg.get("dataset", "dermamnist")).lower()
    aug_mode = cfg["preprocess"].get("augmentation", "standard")
    batch_size = cfg["train"].get("batch_size", 64)
    num_workers = cfg["preprocess"].get("num_workers", 2)

    if dataset_type == "dermamnist":
        resolution = int(cfg["data"].get("resolution", 28))
        upscale_size = int(cfg["preprocess"].get("upscale_size", 64))

        if resolution == 224:
            # Check ~/.medmnist/ first, then local fallback
            home_medmnist = Path.home() / ".medmnist" / "dermamnist_corrected_224.npz"
            alt_medmnist = Path.home() / ".medmnist" / "dermamnist_224.npz"
            if home_medmnist.exists():
                npz_path = home_medmnist
            elif alt_medmnist.exists():
                npz_path = alt_medmnist
            else:
                raise FileNotFoundError(
                    f"DermaMNIST 224x224 file not found at {home_medmnist}. "
                    f"Please place dermamnist_corrected_224.npz in ~/.medmnist/"
                )
        else:
            # Resolution 28
            rel_path = cfg["data"].get("dermamnist_28_path", "DermaMNIST/DermaMNIST_Analysis/NPZ_files/dermamnist_corrected_28.npz")
            npz_path = Path(rel_path)
            if not npz_path.exists():
                # Try finding from workspace root
                npz_path = Path(__file__).resolve().parent.parent.parent.parent / rel_path

        return load_dermamnist_splits(
            npz_path=npz_path,
            resolution=resolution,
            upscale_size=upscale_size,
            augmentation=aug_mode,
            batch_size=batch_size,
            num_workers=num_workers,
        )

    elif dataset_type == "fitzpatrick":
        # Guard: check image completeness
        split_csv_path = Path(cfg["data"].get("split_csv"))
        image_dir = Path(image_dir_override or cfg["data"].get("image_dir", "data/fitzpatrick_images"))

        from vishwas.src.ingest.check_images import check_fitzpatrick_images
        _, _, pct, passes = check_fitzpatrick_images(str(split_csv_path), str(image_dir), min_ratio=0.95)
        if not passes:
            raise RuntimeError(
                f"Training refused: Fitzpatrick local image availability is only {pct:.2f}% (requires >= 95%). "
                f"Extract complete Fitzpatrick17k images into {image_dir.resolve()}"
            )

        df = pd.read_csv(split_csv_path)
        target_col = cfg["preprocess"].get("target_col", "nine_partition_label")
        df = df.dropna(subset=[target_col]).copy()

        unique_labels = sorted(df[target_col].unique())
        label_to_idx = {lbl: idx for idx, lbl in enumerate(unique_labels)}
        idx_to_label = {idx: lbl for lbl, idx in label_to_idx.items()}

        train_df = df[df["partition"] == "train"].reset_index(drop=True)
        val_df = df[df["partition"] == "val"].reset_index(drop=True)
        test_df = df[df["partition"] == "test"].reset_index(drop=True)

        train_tf, eval_tf = get_transforms(aug_mode, target_size=224, is_low_res=False)

        train_ds = FitzpatrickDataset(train_df, image_dir=image_dir, target_col=target_col, label_to_idx=label_to_idx, transform=train_tf)
        val_ds = FitzpatrickDataset(val_df, image_dir=image_dir, target_col=target_col, label_to_idx=label_to_idx, transform=eval_tf)
        test_ds = FitzpatrickDataset(test_df, image_dir=image_dir, target_col=target_col, label_to_idx=label_to_idx, transform=eval_tf)

        dataloaders = {
            "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=vishwas_collate_fn),
            "val": DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=vishwas_collate_fn),
            "test": DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=vishwas_collate_fn),
        }

        info = {
            "dataset_name": "fitzpatrick",
            "has_skin_tone": True,
            "num_classes": len(unique_labels),
            "classes": unique_labels,
            "label_to_idx": label_to_idx,
            "idx_to_label": idx_to_label,
            "train_size": len(train_ds),
            "val_size": len(val_ds),
            "test_size": len(test_ds),
            "target_col": target_col,
            "augmentation": aug_mode,
        }
        return dataloaders, info

    elif dataset_type == "milk10k":
        try:
            from vishwas.src.preprocess.milk10k_dataset import load_milk10k_splits as _load_milk10k
        except ImportError:
            from src.preprocess.milk10k_dataset import load_milk10k_splits as _load_milk10k

        f_csv = Path(cfg["data"].get("milk10k_folds_csv", "data/milk10k_folds.csv"))
        m_dir = Path(image_dir_override or cfg["data"].get("milk10k_dir", "data/milk10k"))
        img_size = int(cfg["preprocess"].get("img_size", 224))
        held_out_fold = int(cfg.get("held_out_fold", 0))
        use_monet = bool(cfg.get("use_monet", False))
        seed = int(cfg.get("project", {}).get("seed", 42))

        return _load_milk10k(
            folds_csv_path=f_csv,
            milk10k_dir=m_dir,
            held_out_fold=held_out_fold,
            val_slice_ratio=0.10,
            img_size=img_size,
            augmentation=aug_mode,
            batch_size=batch_size,
            num_workers=num_workers,
            use_monet=use_monet,
            seed=seed,
        )

    else:
        raise ValueError(f"Unsupported dataset: {dataset_type}. Choose 'milk10k', 'dermamnist', 'fitzpatrick', or 'pad_ufes'.")



if __name__ == "__main__":
    loaders, info = build_dataloaders()
    sample = next(iter(loaders["train"]))
    print(f"Dataset successfully loaded: {info['dataset_name']}")
    print(f"Sample image shape: {sample['image'].shape}")
    print(f"Sample label: {sample['label'][:5]}")
    print(f"Skin-tone group available: {sample['has_skin_tone']}")
