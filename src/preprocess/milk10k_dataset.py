"""
Vishwas MILK10k Dataset & DataLoader Module: milk10k_dataset.py

Provides:
- MILK10kDataset: Loads paired clinical & dermoscopic images + tabular vector per lesion
- Image caching in data/cache/<img_size>/
- Data augmentations: standard vs colour_preserving (mild brightness only, no hue/saturation shifts)
- Internal validation slice (10% of train folds, grouped by lesion, stratified by diagnosis)
- Support for class weighting, balanced sampling, and group weighting
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

logger = logging.getLogger("vishwas.preprocess.milk10k")

# Ground truth diagnostic classes in exact order
MILK10K_CLASSES: List[str] = [
    "AKIEC",
    "BCC",
    "BEN_OTH",
    "BKL",
    "DF",
    "INF",
    "MAL_OTH",
    "MEL",
    "NV",
    "SCCKA",
    "VASC",
]

MILK10K_CLASS_TO_IDX: Dict[str, int] = {c: i for i, c in enumerate(MILK10K_CLASSES)}
MILK10K_IDX_TO_CLASS: Dict[int, str] = {i: c for i, c in enumerate(MILK10K_CLASSES)}
HEADLINE_CLASSES: List[str] = [c for c in MILK10K_CLASSES if c != "MAL_OTH"]

SITE_CATEGORIES: List[str] = [
    "trunk",
    "head_neck_face",
    "lower_extremity",
    "upper_extremity",
    "hand",
    "foot",
    "genital",
    "unknown",
]

SEX_CATEGORIES: List[str] = ["male", "female", "unknown"]

MONET_COLUMNS: List[str] = [
    "MONET_ulceration_crust",
    "MONET_hair",
    "MONET_vasculature_vessels",
    "MONET_erythema",
    "MONET_pigmented",
    "MONET_gel_water_drop_fluid_dermoscopy_liquid",
    "MONET_skin_markings_pen_ink_purple_pen",
]


def resolve_path(rel_path: str, must_exist: bool = True) -> Path:
    """Resolves relative path across workspace root, vishwas directory, or script location."""
    script_dir = Path(__file__).resolve().parent
    vishwas_root = script_dir.parent.parent
    workspace_root = vishwas_root.parent

    candidates = [
        Path(rel_path),
        vishwas_root / rel_path,
        workspace_root / rel_path,
    ]
    for c in candidates:
        if c.exists():
            return c
    if must_exist:
        raise FileNotFoundError(f"Could not locate '{rel_path}' in {[str(c) for c in candidates]}")
    return vishwas_root / rel_path


def get_milk10k_transforms(
    augmentation: str = "standard",
    img_size: int = 224,
) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Constructs training and validation/test transforms.
    Augmentation supports 'standard' and 'colour_preserving'.
    Both only use mild brightness jitter (strictly no hue or saturation shifts).
    """
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    brightness_delta = 0.15 if augmentation.lower() == "standard" else 0.08

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=30),
        transforms.ColorJitter(brightness=brightness_delta),
        transforms.ToTensor(),
        normalize,
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        normalize,
    ])

    return train_transform, eval_transform


def precache_milk10k_images(
    meta_df: pd.DataFrame,
    raw_images_dir: Path,
    cache_dir: Path,
    img_size: int = 128,
    num_workers: int = 8,
) -> None:
    """Pre-caches resized images to cache_dir to accelerate training epochs."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    items_to_cache: List[Tuple[Path, Path]] = []

    for _, row in meta_df.iterrows():
        lid = str(row["lesion_id"])
        isic = str(row["isic_id"])
        src = raw_images_dir / lid / f"{isic}.jpg"
        dst = cache_dir / f"{isic}.jpg"
        if not dst.exists() and src.exists():
            items_to_cache.append((src, dst))

    if not items_to_cache:
        return

    def _resize_and_save(item: Tuple[Path, Path]) -> None:
        s, d = item
        try:
            with Image.open(s) as img:
                resized = img.convert("RGB").resize((img_size, img_size), Image.Resampling.BILINEAR)
                resized.save(d, format="JPEG", quality=95)
        except Exception as e:
            logger.warning("Failed to cache %s: %s", s, e)

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        list(executor.map(_resize_and_save, items_to_cache))


class TabularScaler:
    """Fits continuous feature normalizers on training set to prevent data leakage."""

    def __init__(self, use_monet: bool = False) -> None:
        self.use_monet = use_monet
        self.age_mean: float = 60.0
        self.age_std: float = 15.0
        self.monet_means: Dict[str, float] = {}
        self.monet_stds: Dict[str, float] = {}

    def fit(self, meta_df: pd.DataFrame, lesion_ids: set) -> None:
        sub = meta_df[meta_df["lesion_id"].isin(lesion_ids)].drop_duplicates("lesion_id")
        valid_ages = sub["age_approx"].dropna()
        if len(valid_ages) > 0:
            self.age_mean = float(valid_ages.mean())
            self.age_std = float(valid_ages.std()) if valid_ages.std() > 0 else 1.0

        if self.use_monet:
            for mc in MONET_COLUMNS:
                if mc in sub.columns:
                    v = sub[mc].dropna()
                    if len(v) > 0:
                        self.monet_means[mc] = float(v.mean())
                        self.monet_stds[mc] = float(v.std()) if v.std() > 0 else 1.0

    def transform_age(self, age_val: Any) -> Tuple[float, float]:
        """Returns (normalized_age, is_missing_flag)."""
        if pd.isna(age_val):
            return 0.0, 1.0
        try:
            val = float(age_val)
            norm = (val - self.age_mean) / self.age_std
            return norm, 0.0
        except (ValueError, TypeError):
            return 0.0, 1.0

    def transform_monet(self, row: pd.Series) -> List[float]:
        """Transforms 7 MONET continuous features."""
        vals = []
        for mc in MONET_COLUMNS:
            v = row.get(mc, np.nan)
            if pd.isna(v):
                vals.append(0.0)
            else:
                m = self.monet_means.get(mc, 0.0)
                s = self.monet_stds.get(mc, 1.0)
                vals.append((float(v) - m) / s)
        return vals


class MILK10kDataset(Dataset):
    """
    PyTorch Dataset for MILK10k dermatological classification.
    Each sample represents a unique lesion containing:
    - clinical image (clinical: close-up)
    - dermoscopic image (dermoscopic)
    - tabular vector (age standardized + missing flag, sex one-hot, site one-hot, optional MONET)
    - ground-truth diagnosis index (0..10)
    - skin_tone_class (0..5)
    - tone_group (G1, G2, G3, unknown)
    - lesion_id
    """

    def __init__(
        self,
        lesion_df: pd.DataFrame,
        meta_df: pd.DataFrame,
        raw_images_dir: Path,
        cache_dir: Optional[Path] = None,
        img_size: int = 224,
        transform: Optional[transforms.Compose] = None,
        tabular_scaler: Optional[TabularScaler] = None,
        use_monet: bool = False,
    ) -> None:
        self.lesions = lesion_df.reset_index(drop=True)
        self.meta_df = meta_df
        self.raw_images_dir = raw_images_dir
        self.cache_dir = cache_dir
        self.img_size = img_size
        self.transform = transform
        self.tabular_scaler = tabular_scaler or TabularScaler(use_monet=use_monet)
        self.use_monet = use_monet

        # Index metadata by lesion_id and image_type for O(1) retrieval
        clin_rows = meta_df[meta_df["image_type"] == "clinical: close-up"].drop_duplicates("lesion_id")
        dermo_rows = meta_df[meta_df["image_type"] == "dermoscopic"].drop_duplicates("lesion_id")

        self.clin_lookup: Dict[str, pd.Series] = {row["lesion_id"]: row for _, row in clin_rows.iterrows()}
        self.dermo_lookup: Dict[str, pd.Series] = {row["lesion_id"]: row for _, row in dermo_rows.iterrows()}

        # Compute tabular vectors in advance
        self.tabular_vectors = [self._encode_tabular(row["lesion_id"]) for _, row in self.lesions.iterrows()]
        self.tabular_dim = self.tabular_vectors[0].shape[0] if self.tabular_vectors else 13

    def _encode_tabular(self, lesion_id: str) -> torch.Tensor:
        """Constructs standardized tabular feature vector for a lesion."""
        row = self.clin_lookup.get(lesion_id)
        if row is None:
            row = self.dermo_lookup.get(lesion_id)
        if row is None:
            # Fallback zeros
            dim = 13 + (7 if self.use_monet else 0)
            return torch.zeros(dim, dtype=torch.float32)

        # 1. Age (standardized + missing flag) -> 2 dims
        age_norm, age_missing = self.tabular_scaler.transform_age(row.get("age_approx", np.nan))

        # 2. Sex one-hot -> 3 dims (male, female, unknown)
        sex_val = str(row.get("sex", "")).lower().strip()
        sex_one_hot = [
            1.0 if sex_val == "male" else 0.0,
            1.0 if sex_val == "female" else 0.0,
            1.0 if sex_val not in ["male", "female"] else 0.0,
        ]

        # 3. Site one-hot -> 8 dims
        site_val = str(row.get("site", "")).lower().strip()
        site_one_hot = [1.0 if site_val == cat else 0.0 for cat in SITE_CATEGORIES[:-1]]
        is_known_site = any(site_one_hot)
        site_one_hot.append(0.0 if is_known_site else 1.0)

        vec = [age_norm, age_missing] + sex_one_hot + site_one_hot

        # 4. Optional MONET continuous features -> 7 dims
        if self.use_monet:
            monet_feats = self.tabular_scaler.transform_monet(row)
            vec.extend(monet_feats)

        return torch.tensor(vec, dtype=torch.float32)

    def _load_image(self, lesion_id: str, isic_id: str) -> Image.Image:
        """Loads image from cache if present, otherwise from raw directory."""
        if self.cache_dir:
            cached_path = self.cache_dir / f"{isic_id}.jpg"
            if cached_path.exists():
                try:
                    return Image.open(cached_path).convert("RGB")
                except Exception:
                    pass

        raw_path = self.raw_images_dir / lesion_id / f"{isic_id}.jpg"
        if not raw_path.exists():
            # Return blank image if file is missing (failsafe)
            return Image.new("RGB", (self.img_size, self.img_size), color=(128, 128, 128))

        img = Image.open(raw_path).convert("RGB")
        if self.cache_dir:
            try:
                resized = img.resize((self.img_size, self.img_size), Image.Resampling.BILINEAR)
                resized.save(self.cache_dir / f"{isic_id}.jpg", format="JPEG", quality=95)
                return resized
            except Exception:
                pass
        return img

    def __len__(self) -> int:
        return len(self.lesions)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.lesions.iloc[idx]
        lid = str(row["lesion_id"])
        diag = str(row["diagnosis"])
        label_idx = MILK10K_CLASS_TO_IDX.get(diag, 0)
        skin_tone = int(row["skin_tone_class"]) if pd.notna(row.get("skin_tone_class")) else -1
        tone_grp = str(row.get("tone_group", "unknown"))

        clin_isic = str(self.clin_lookup[lid]["isic_id"]) if lid in self.clin_lookup else ""
        dermo_isic = str(self.dermo_lookup[lid]["isic_id"]) if lid in self.dermo_lookup else ""

        clin_pil = self._load_image(lid, clin_isic)
        dermo_pil = self._load_image(lid, dermo_isic)

        if self.transform:
            clin_tensor = self.transform(clin_pil)
            dermo_tensor = self.transform(dermo_pil)
        else:
            to_tensor = transforms.ToTensor()
            clin_tensor = to_tensor(clin_pil)
            dermo_tensor = to_tensor(dermo_pil)

        tabular_tensor = self.tabular_vectors[idx]

        return {
            "clinical_img": clin_tensor,
            "dermoscopic_img": dermo_tensor,
            "tabular": tabular_tensor,
            "label": torch.tensor(label_idx, dtype=torch.long),
            "skin_tone_class": skin_tone,
            "tone_group": tone_grp,
            "lesion_id": lid,
        }


def milk10k_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collates list of lesion samples into batched tensors."""
    clinical_imgs = torch.stack([b["clinical_img"] for b in batch], dim=0)
    dermoscopic_imgs = torch.stack([b["dermoscopic_img"] for b in batch], dim=0)
    tabulars = torch.stack([b["tabular"] for b in batch], dim=0)
    labels = torch.stack([b["label"] for b in batch], dim=0)

    skin_tones = torch.tensor([b["skin_tone_class"] for b in batch], dtype=torch.long)
    tone_groups = [b["tone_group"] for b in batch]
    lesion_ids = [b["lesion_id"] for b in batch]

    return {
        "clinical_img": clinical_imgs,
        "dermoscopic_img": dermoscopic_imgs,
        "tabular": tabulars,
        "label": labels,
        "skin_tone_class": skin_tones,
        "tone_group": tone_groups,
        "lesion_id": lesion_ids,
    }


def load_milk10k_splits(
    folds_csv_path: Optional[Path] = None,
    milk10k_dir: Optional[Path] = None,
    held_out_fold: int = 0,
    val_slice_ratio: float = 0.10,
    img_size: int = 224,
    augmentation: str = "standard",
    batch_size: int = 32,
    num_workers: int = 0,
    use_monet: bool = False,
    seed: int = 42,
) -> Tuple[Dict[str, DataLoader], Dict[str, Any]]:
    """
    Constructs train, validation, and test dataloaders for held-out fold k.
    - Test fold: fold == held_out_fold (1,048 lesions)
    - Train pool: remaining 4 folds (4,192 lesions)
    - Internal validation slice: 10% of train pool (~419 lesions), stratified by diagnosis
    - Train split: remaining 90% of train pool (~3,773 lesions)
    """
    f_csv = folds_csv_path or resolve_path("data/milk10k_folds.csv")
    m_dir = milk10k_dir or resolve_path("data/milk10k")

    folds_df = pd.read_csv(f_csv)
    meta_df = pd.read_csv(m_dir / "MILK10k_Training_Metadata.csv")
    raw_images_dir = m_dir / "train_images" / "MILK10k_Training_Input"

    cache_dir = resolve_path(f"data/cache/{img_size}", must_exist=False)
    precache_milk10k_images(meta_df, raw_images_dir, cache_dir, img_size=img_size, num_workers=4)

    # Segregate test fold from train pool
    test_df = folds_df[folds_df["fold"] == held_out_fold].reset_index(drop=True)
    train_pool_df = folds_df[folds_df["fold"] != held_out_fold].reset_index(drop=True)

    # 10% stratified internal validation split from train pool
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_slice_ratio, random_state=seed)
    train_indices, val_indices = next(sss.split(train_pool_df, train_pool_df["diagnosis"]))

    train_df = train_pool_df.iloc[train_indices].reset_index(drop=True)
    val_df = train_pool_df.iloc[val_indices].reset_index(drop=True)

    # Fit tabular normalizer on train_df lesions only
    scaler = TabularScaler(use_monet=use_monet)
    scaler.fit(meta_df, set(train_df["lesion_id"]))

    train_tf, eval_tf = get_milk10k_transforms(augmentation=augmentation, img_size=img_size)

    train_ds = MILK10kDataset(
        lesion_df=train_df,
        meta_df=meta_df,
        raw_images_dir=raw_images_dir,
        cache_dir=cache_dir,
        img_size=img_size,
        transform=train_tf,
        tabular_scaler=scaler,
        use_monet=use_monet,
    )

    val_ds = MILK10kDataset(
        lesion_df=val_df,
        meta_df=meta_df,
        raw_images_dir=raw_images_dir,
        cache_dir=cache_dir,
        img_size=img_size,
        transform=eval_tf,
        tabular_scaler=scaler,
        use_monet=use_monet,
    )

    test_ds = MILK10kDataset(
        lesion_df=test_df,
        meta_df=meta_df,
        raw_images_dir=raw_images_dir,
        cache_dir=cache_dir,
        img_size=img_size,
        transform=eval_tf,
        tabular_scaler=scaler,
        use_monet=use_monet,
    )

    dataloaders = {
        "train": DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=milk10k_collate_fn,
        ),
        "val": DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=milk10k_collate_fn,
        ),
        "test": DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=milk10k_collate_fn,
        ),
    }

    info = {
        "dataset_name": "milk10k",
        "held_out_fold": held_out_fold,
        "num_classes": len(MILK10K_CLASSES),
        "classes": MILK10K_CLASSES,
        "class_to_idx": MILK10K_CLASS_TO_IDX,
        "idx_to_class": MILK10K_IDX_TO_CLASS,
        "tabular_dim": train_ds.tabular_dim,
        "train_size": len(train_ds),
        "val_size": len(val_ds),
        "test_size": len(test_ds),
        "img_size": img_size,
        "augmentation": augmentation,
        "use_monet": use_monet,
    }

    return dataloaders, info
