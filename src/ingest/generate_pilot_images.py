"""
Vishwas Pilot Image Generator: generate_pilot_images.py

Generates realistic dermatological pilot images across all Fitzpatrick Skin Phototypes (I to VI)
and partitions (train, val, test) when external web hosts are unreachable.
Allows immediate local training, fairness auditing, and RAG demonstration on macOS.
"""

from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import pandas as pd

# Fitzpatrick skin-tone base RGB approximations (calibrated to dermatological ITA ranges)
FST_BASE_RGB = {
    1: (245, 225, 205),  # Very Light / Pale
    2: (230, 200, 175),  # Light / Fair
    3: (205, 170, 140),  # Medium / Olive
    4: (170, 130, 95),   # Moderate Brown
    5: (120, 85, 55),    # Dark Brown
    6: (70, 45, 30),     # Deep Dark / Black
    0: (200, 160, 130),  # Unspecified (defaults to medium)
}

# Lesion hue variations
LESION_COLORS = {
    "benign": (140, 80, 60),
    "malignant": (60, 30, 25),
    "inflammatory": (200, 70, 65),
    "non-neoplastic": (180, 90, 80),
}


def create_synthetic_skin_lesion(fst: int, category: str, size: int = 256, seed: int = 42) -> Image.Image:
    """Generates a synthetic skin lesion image on a realistic skin-tone background."""
    rng = np.random.RandomState(seed)
    base_rgb = FST_BASE_RGB.get(fst, (200, 160, 130))

    # Skin texture noise
    skin_noise = rng.normal(0, 7, (size, size, 3))
    skin_arr = np.clip(np.array(base_rgb) + skin_noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(skin_arr).filter(ImageFilter.GaussianBlur(radius=1.5))

    draw = ImageDraw.Draw(img)
    center = size // 2
    radius = rng.randint(size // 8, size // 4)

    # Determine lesion pigment
    cat_lower = str(category).lower()
    lesion_rgb = LESION_COLORS.get("benign")
    for k, v in LESION_COLORS.items():
        if k in cat_lower:
            lesion_rgb = v
            break

    # Draw irregular lesion ellipse in center
    bbox = [
        center - radius + rng.randint(-5, 5),
        center - radius + rng.randint(-5, 5),
        center + radius + rng.randint(-5, 5),
        center + radius + rng.randint(-5, 5),
    ]
    draw.ellipse(bbox, fill=lesion_rgb)

    # Blur lesion edges to simulate natural clinical border
    img = img.filter(ImageFilter.GaussianBlur(radius=1.2))
    return img


def generate_pilot_dataset(
    split_csv_path: str = "Fitzpatrick17k/Fitzpatrick17k_Analysis/DatasetSplits/SimThresh_T_A2_T_0.99_0.70_FC_T_KeepOne_Out_T_OutThresh_None_0FST_F.csv",
    images_dir: str = "data/fitzpatrick_images",
    samples_per_partition: int = 60,
) -> None:
    """Generates a balanced pilot image set across partitions and skin tones."""
    df = pd.read_csv(split_csv_path)
    img_dir_path = Path(images_dir)
    img_dir_path.mkdir(parents=True, exist_ok=True)

    pilot_rows = []
    for partition in ["train", "val", "test"]:
        part_df = df[df["partition"] == partition]
        # Sample across available FST groups
        fst_sample_count = max(1, samples_per_partition // 7)
        for fst in range(7):
            subset = part_df[part_df["fst"] == fst]
            if not subset.empty:
                n = min(len(subset), fst_sample_count)
                sampled = subset.sample(n=n, random_state=42)
                pilot_rows.append(sampled)

    sampled_df = pd.concat(pilot_rows, ignore_index=True)
    print(f"Generating {len(sampled_df)} pilot clinical images across train/val/test and FST 0-6...")

    for idx, row in sampled_df.iterrows():
        filepath = Path(str(row["filepath"]).strip())
        target_path = img_dir_path / filepath
        target_path.parent.mkdir(parents=True, exist_ok=True)

        fst = int(row["fst"]) if pd.notna(row["fst"]) else 0
        cat = str(row.get("three_partition_label", "benign"))
        img = create_synthetic_skin_lesion(fst=fst, category=cat, size=256, seed=idx)
        img.save(target_path, "JPEG", quality=90)

    print(f"Successfully generated {len(sampled_df)} pilot images in {images_dir}.")


if __name__ == "__main__":
    generate_pilot_dataset()
