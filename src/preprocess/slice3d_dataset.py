"""
Vishwas SLICE-3D Dataset & DataLoader Module: slice3d_dataset.py

Provides:
- Slice3DDataset: PyTorch Dataset for loading SLICE-3D lesion crops
- Color and spatial transforms consistent with MILK10k (ImageNet normalization, mild brightness jitter)
- Balanced resampling option: all malignant + benign_per_malignant per epoch
- Positive class weighting computation for BCE loss
- DataLoader construction for train, val, and test partitions
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision import transforms

logger = logging.getLogger("vishwas.preprocess.slice3d")

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent.parent
VISHWAS_ROOT = Path(__file__).resolve().parent.parent.parent


def resolve_path(rel_path: str, must_exist: bool = True) -> Path:
    """Resolves relative path across workspace root, vishwas directory, or cwd."""
    candidates = [
        Path(rel_path),
        VISHWAS_ROOT / rel_path,
        WORKSPACE_ROOT / rel_path,
        Path.cwd() / rel_path,
    ]
    for c in candidates:
        if c.exists():
            return c.resolve()
    if must_exist:
        raise FileNotFoundError(f"Could not locate '{rel_path}' in {[str(c) for c in candidates]}")
    return (VISHWAS_ROOT / rel_path).resolve()


def get_slice3d_transforms(img_size: int = 128) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Constructs train and eval transforms.
    Consistent with milk10k_dataset.py:
    - ImageNet normalization
    - RandomResizedCrop, horizontal/vertical flips, mild rotation, mild brightness jitter only
    - No hue or saturation shifts (preserves melanin/erythema diagnostic colorimetry)
    """
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=30),
        transforms.ColorJitter(brightness=0.15),
        transforms.ToTensor(),
        normalize,
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        normalize,
    ])

    return train_transform, eval_transform


class Slice3DDataset(Dataset):
    """
    PyTorch Dataset for SLICE-3D binary lesion crops.
    Returns:
        image: Transformed image tensor (3, img_size, img_size)
        target: Float tensor (0.0 or 1.0)
        isic_id: ISIC image identifier string
        patient_id: Patient identifier string
    """

    def __init__(
        self,
        df: pd.DataFrame,
        images_dir: Path,
        transform: Optional[Callable] = None,
        img_size: int = 128,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.images_dir = images_dir
        self.transform = transform
        self.img_size = img_size

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        isic_id = str(row["isic_id"])
        target_val = float(row["target"])
        patient_id = str(row["patient_id"])

        img_path = self.images_dir / f"{isic_id}.jpg"
        if img_path.exists():
            try:
                img = Image.open(img_path).convert("RGB")
            except Exception as e:
                logger.warning(f"Failed to open {img_path}: {e}. Using blank.")
                img = Image.new("RGB", (self.img_size, self.img_size), color=(128, 128, 128))
        else:
            img = Image.new("RGB", (self.img_size, self.img_size), color=(128, 128, 128))

        if self.transform:
            img_tensor = self.transform(img)
        else:
            to_tensor = transforms.ToTensor()
            img_tensor = to_tensor(img)

        return {
            "image": img_tensor,
            "target": torch.tensor(target_val, dtype=torch.float32),
            "isic_id": isic_id,
            "patient_id": patient_id,
        }


class BalancedEpochSampler(Sampler[int]):
    """
    Epoch-wise sampler: includes ALL malignant samples + (benign_per_malignant * n_malignant)
    randomly selected benign samples, resampled every epoch.
    """

    def __init__(
        self,
        targets: np.ndarray,
        benign_per_malignant: int = 20,
        seed: int = 42,
    ) -> None:
        self.targets = np.asarray(targets)
        self.mal_indices = np.where(self.targets == 1)[0]
        self.ben_indices = np.where(self.targets == 0)[0]
        self.benign_per_malignant = benign_per_malignant
        self.rng = np.random.RandomState(seed)
        self.current_indices: List[int] = []
        self.resample()

    def resample(self) -> None:
        n_needed_benign = min(len(self.ben_indices), len(self.mal_indices) * self.benign_per_malignant)
        sampled_ben = self.rng.choice(self.ben_indices, size=n_needed_benign, replace=False)
        combined = np.concatenate([self.mal_indices, sampled_ben])
        self.rng.shuffle(combined)
        self.current_indices = combined.tolist()

    def set_epoch(self, epoch: int) -> None:
        self.rng = np.random.RandomState(42 + epoch)
        self.resample()

    def __iter__(self):
        return iter(self.current_indices)

    def __len__(self) -> int:
        return len(self.current_indices)


def compute_slice3d_pos_weight(df: pd.DataFrame) -> torch.Tensor:
    """Computes pos_weight = n_benign / n_malignant for BCEWithLogitsLoss."""
    targets = df["target"].to_numpy()
    n_pos = int(np.sum(targets == 1))
    n_neg = int(np.sum(targets == 0))
    if n_pos == 0:
        return torch.tensor([1.0], dtype=torch.float32)
    weight = float(n_neg) / float(n_pos)
    return torch.tensor([weight], dtype=torch.float32)


def build_slice3d_dataloaders(
    split_csv_path: Optional[Path] = None,
    images_dir: Optional[Path] = None,
    img_size: int = 128,
    batch_size: int = 64,
    num_workers: int = 2,
    balanced: bool = False,
    benign_per_malignant: int = 20,
    max_images: Optional[int] = None,
    seed: int = 42,
) -> Tuple[Dict[str, DataLoader], Dict[str, Any]]:
    """
    Constructs train, val, and test DataLoaders from the 100k split CSV.
    If max_images is set, slices the dataset proportionally for fast smoke runs.
    """
    csv_path = split_csv_path or resolve_path("data/slice3d_split_100k.csv")
    img_dir = images_dir or resolve_path("data/slice3d/images/ISIC_2024_Training_Input")

    full_split_df = pd.read_csv(csv_path)

    train_df = full_split_df[full_split_df["split"] == "train"].copy().reset_index(drop=True)
    val_df = full_split_df[full_split_df["split"] == "val"].copy().reset_index(drop=True)
    test_df = full_split_df[full_split_df["split"] == "test"].copy().reset_index(drop=True)

    if max_images is not None and max_images > 0 and max_images < len(full_split_df):
        # Scale train/val/test splits preserving all malignant samples in the slice
        frac = float(max_images) / float(len(full_split_df))
        n_tr = max(200, int(len(train_df) * frac))
        n_va = max(50, int(len(val_df) * frac))
        n_te = max(50, int(len(test_df) * frac))

        def _sample_with_mal(split_sub_df: pd.DataFrame, n_target: int) -> pd.DataFrame:
            mal = split_sub_df[split_sub_df["target"] == 1]
            ben = split_sub_df[split_sub_df["target"] == 0]
            n_ben = max(1, n_target - len(mal))
            ben_s = ben.sample(n=min(len(ben), n_ben), random_state=seed)
            return pd.concat([mal, ben_s]).sample(frac=1.0, random_state=seed).reset_index(drop=True)

        train_df = _sample_with_mal(train_df, n_tr)
        val_df = _sample_with_mal(val_df, n_va)
        test_df = _sample_with_mal(test_df, n_te)
        logger.info(f"Subsampled max_images={max_images} -> train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}")

    train_tf, eval_tf = get_slice3d_transforms(img_size=img_size)

    train_ds = Slice3DDataset(train_df, img_dir, transform=train_tf, img_size=img_size)
    val_ds = Slice3DDataset(val_df, img_dir, transform=eval_tf, img_size=img_size)
    test_ds = Slice3DDataset(test_df, img_dir, transform=eval_tf, img_size=img_size)

    sampler = None
    if balanced:
        sampler = BalancedEpochSampler(
            targets=train_df["target"].to_numpy(),
            benign_per_malignant=benign_per_malignant,
            seed=seed,
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
        )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    pos_weight = compute_slice3d_pos_weight(train_df)

    metadata = {
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
        "pos_weight": pos_weight,
        "sampler": sampler,
        "balanced": balanced,
        "img_size": img_size,
    }

    return {"train": train_loader, "val": val_loader, "test": test_loader}, metadata
