"""
Vishwas Dataset Ingest Module: download_fitzpatrick.py

Downloads and validates images for the Fitzpatrick17k-C dataset according to the
official cleaned splits. Includes multithreading, per-domain rate limiting, exponential
backoff retries, PIL verification, resumability, and comprehensive summary reporting.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import pandas as pd
import requests
import yaml
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("vishwas.ingest")


class DomainRateLimiter:
    """Thread-safe per-domain rate limiter to avoid overloading host servers."""

    def __init__(self, min_interval_seconds: float = 0.1) -> None:
        self.min_interval = min_interval_seconds
        self.last_access: Dict[str, float] = defaultdict(float)
        self.lock = Lock()

    def wait(self, domain: str) -> None:
        with self.lock:
            now = time.time()
            elapsed = now - self.last_access[domain]
            if elapsed < self.min_interval:
                time.sleep(self.min_interval - elapsed)
            self.last_access[domain] = time.time()


def extract_domain(url: Optional[str]) -> str:
    """Extracts netloc domain from a URL or returns 'unknown'."""
    if not url or not isinstance(url, str) or not url.strip():
        return "unknown"
    try:
        netloc = urlparse(url.strip()).netloc
        return netloc if netloc else "unknown"
    except Exception:
        return "unknown"


def validate_image_file(file_path: Path) -> bool:
    """
    Validates whether an image file can be opened and parsed by PIL.
    Returns True if valid, False if corrupt.
    """
    try:
        with Image.open(file_path) as img:
            img.verify()
        # Verify only reads headers; re-open and load a small chunk or load fully
        with Image.open(file_path) as img:
            img.load()
        return True
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError):
        return False


def download_single_image(
    row: Dict[str, Any],
    images_dir: Path,
    session: requests.Session,
    rate_limiter: DomainRateLimiter,
    user_agent: str,
    timeout: int = 20,
    max_retries: int = 3,
    backoff_factor: float = 1.5,
) -> Tuple[bool, Dict[str, Any], Optional[Dict[str, Any]]]:
    """
    Downloads and verifies a single image file.

    Returns:
        (is_success, clean_record_dict, failed_record_dict)
    """
    filepath = str(row.get("filepath", "")).strip()
    url = str(row.get("url", "")).strip() if pd.notna(row.get("url")) else ""
    md5hash = str(row.get("md5hash", "")).strip()
    domain = extract_domain(url)

    target_path = images_dir / filepath

    # 1. Check if already downloaded and valid (Resumability)
    if target_path.exists():
        if validate_image_file(target_path):
            clean_record = {
                "image_path": str(target_path),
                "md5hash": md5hash,
                "label": row.get("label"),
                "diag": row.get("diag"),
                "nine_partition_label": row.get("nine_partition_label"),
                "three_partition_label": row.get("three_partition_label"),
                "fst": row.get("fst"),
                "fitzpatrick": row.get("fitzpatrick"),
                "partition": row.get("partition"),
                "qc": row.get("qc"),
                "domain": domain,
            }
            return True, clean_record, None
        else:
            # Corrupt existing file; delete and re-attempt
            try:
                os.remove(target_path)
            except OSError:
                pass

    # 2. Check for missing or invalid URL
    if not url or url == "nan" or not url.startswith("http"):
        failed_record = {
            "md5hash": md5hash,
            "url": url,
            "error": "Missing or invalid URL",
            "domain": domain,
        }
        return False, {}, failed_record

    # 3. Create parent directories
    target_path.parent.mkdir(parents=True, exist_ok=True)

    headers = {
        "User-Agent": user_agent,
        "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
    }

    last_error = "Unknown error"

    # 4. Attempt download with retries and exponential backoff
    for attempt in range(1, max_retries + 1):
        rate_limiter.wait(domain)
        try:
            response = session.get(
                url,
                headers=headers,
                timeout=timeout,
                allow_redirects=True,
                stream=True,
            )

            if response.status_code == 200:
                temp_path = target_path.with_suffix(".tmp")
                with open(temp_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=16384):
                        if chunk:
                            f.write(chunk)

                # Validate with PIL
                if validate_image_file(temp_path):
                    if temp_path.exists():
                        temp_path.replace(target_path)
                    clean_record = {
                        "image_path": str(target_path),
                        "md5hash": md5hash,
                        "label": row.get("label"),
                        "diag": row.get("diag"),
                        "nine_partition_label": row.get("nine_partition_label"),
                        "three_partition_label": row.get("three_partition_label"),
                        "fst": row.get("fst"),
                        "fitzpatrick": row.get("fitzpatrick"),
                        "partition": row.get("partition"),
                        "qc": row.get("qc"),
                        "domain": domain,
                    }
                    return True, clean_record, None
                else:
                    if temp_path.exists():
                        os.remove(temp_path)
                    last_error = "Corrupted image file / unidentifiable by PIL"
            else:
                last_error = f"HTTP {response.status_code}"

        except requests.exceptions.Timeout:
            last_error = f"Timeout ({timeout}s)"
        except requests.exceptions.ConnectionError as e:
            last_error = f"Connection error: {type(e).__name__}"
        except Exception as e:
            last_error = f"Error: {str(e)}"

        # Exponential backoff before retry
        if attempt < max_retries:
            time.sleep(backoff_factor**attempt)

    failed_record = {
        "md5hash": md5hash,
        "url": url,
        "error": last_error,
        "domain": domain,
    }
    return False, {}, failed_record


def print_dataset_statistics(df: pd.DataFrame) -> None:
    """Prints comprehensive statistics of the input split CSV."""
    print("\n" + "=" * 70)
    print("           FITZPATRICK17K-C SPLIT DATASET OVERVIEW")
    print("=" * 70)
    print(f"Total rows in split CSV: {len(df)}")

    print("\n--- Value Counts: Partition ---")
    print(df["partition"].value_counts(dropna=False).to_string())

    print("\n--- Value Counts: FST (Skin-Tone Grouping) ---")
    print(df["fst"].value_counts(dropna=False).sort_index().to_string())

    print("\n--- Value Counts: Fitzpatrick (Original Annotations, -1 = missing) ---")
    print(df["fitzpatrick"].value_counts(dropna=False).sort_index().to_string())

    print("\n--- Value Counts: QC (Quality Control Annotations) ---")
    print(df["qc"].value_counts(dropna=False).to_string())

    n_labels = df["label"].nunique() if "label" in df.columns else 0
    n_nine = df["nine_partition_label"].nunique() if "nine_partition_label" in df.columns else 0
    n_three = df["three_partition_label"].nunique() if "three_partition_label" in df.columns else 0

    print("\n--- Unique Label Counts ---")
    print(f"Detailed disease labels (label):        {n_labels}")
    print(f"Nine-partition labels:                  {n_nine}")
    print(f"Three-partition labels (benign/mal/inf): {n_three}")

    diff_fst_fitz = (df["fst"] != df["fitzpatrick"]).sum()
    print("\n--- Annotation Consistency Check ---")
    print(f"Number of rows where fst != fitzpatrick: {diff_fst_fitz}")
    if diff_fst_fitz > 0:
        print("Breakdown of differences (fst vs fitzpatrick):")
        diff_df = df[df["fst"] != df["fitzpatrick"]]
        print(diff_df.groupby(["fst", "fitzpatrick"]).size().to_string())
    print("=" * 70 + "\n")


def print_download_summary(
    clean_df: pd.DataFrame,
    failed_df: pd.DataFrame,
    all_domains: List[str],
) -> None:
    """Prints summary of download attempts, successes, failures, and train label distributions."""
    print("\n" + "=" * 70)
    print("           INGESTION & DOWNLOAD RESULTS SUMMARY")
    print("=" * 70)

    # Domain breakdown
    domain_stats: Dict[str, Dict[str, int]] = defaultdict(lambda: {"attempted": 0, "succeeded": 0, "failed": 0})
    for d in all_domains:
        domain_stats[d]["attempted"] += 1

    if not clean_df.empty and "domain" in clean_df.columns:
        for d, count in clean_df["domain"].value_counts().items():
            domain_stats[d]["succeeded"] += count

    if not failed_df.empty and "domain" in failed_df.columns:
        for d, count in failed_df["domain"].value_counts().items():
            domain_stats[d]["failed"] += count

    print(f"{'Domain':<35} | {'Attempted':<10} | {'Succeeded':<10} | {'Failed':<10}")
    print("-" * 73)
    for dom, s in sorted(domain_stats.items(), key=lambda x: x[1]["attempted"], reverse=True):
        print(f"{dom:<35} | {s['attempted']:<10} | {s['succeeded']:<10} | {s['failed']:<10}")

    print("\n--- Successful Downloads: Partition Distribution ---")
    if not clean_df.empty and "partition" in clean_df.columns:
        print(clean_df["partition"].value_counts(dropna=False).to_string())
    else:
        print("No images downloaded yet.")

    print("\n--- Successful Downloads: FST Distribution ---")
    if not clean_df.empty and "fst" in clean_df.columns:
        print(clean_df["fst"].value_counts(dropna=False).sort_index().to_string())
    else:
        print("No images downloaded yet.")

    print("\n--- Rare Class Flagging (< 10 images in 'train' partition) ---")
    if not clean_df.empty and "partition" in clean_df.columns and "label" in clean_df.columns:
        train_df = clean_df[clean_df["partition"] == "train"]
        train_label_counts = train_df["label"].value_counts()
        rare_labels = train_label_counts[train_label_counts < 10]
        if len(rare_labels) > 0:
            print(f"Warning: {len(rare_labels)} labels have fewer than 10 training images:")
            for lbl, cnt in rare_labels.items():
                print(f"  - {lbl}: {cnt} train images")
        else:
            print("All downloaded labels have at least 10 training images.")
    else:
        print("No train images available to check.")

    print("=" * 70 + "\n")


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Loads configuration from yaml file."""
    if config_path and Path(config_path).exists():
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    default_config = Path("vishwas/configs/config.yaml")
    if default_config.exists():
        with open(default_config, "r") as f:
            return yaml.safe_load(f)
    return {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Vishwas Fitzpatrick17k-C Image Ingest & Download")
    parser.add_argument("--config", type=str, default="vishwas/configs/config.yaml", help="Path to config.yaml")
    parser.add_argument("--split-csv", type=str, default=None, help="Path to split CSV")
    parser.add_argument("--images-dir", type=str, default=None, help="Directory to save downloaded images")
    parser.add_argument("--clean-metadata-csv", type=str, default=None, help="Output clean metadata CSV")
    parser.add_argument("--failed-downloads-csv", type=str, default=None, help="Output failed downloads CSV")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of images to attempt")
    parser.add_argument("--workers", type=int, default=None, help="Number of download threads")
    parser.add_argument("--timeout", type=int, default=None, help="Request timeout in seconds")
    parser.add_argument("--sync-existing", action="store_true", help="Sync all existing verified local images directly into clean metadata")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_cfg = cfg.get("data", {})
    ingest_cfg = cfg.get("ingest", {})

    split_csv_path = Path(
        args.split_csv
        or data_cfg.get(
            "split_csv",
            "Fitzpatrick17k/Fitzpatrick17k_Analysis/DatasetSplits/SimThresh_T_A2_T_0.99_0.70_FC_T_KeepOne_Out_T_OutThresh_None_0FST_F.csv",
        )
    )
    images_dir = Path(args.images_dir or data_cfg.get("images_dir", "data/fitzpatrick_images"))
    clean_csv_path = Path(args.clean_metadata_csv or data_cfg.get("clean_metadata_csv", "data/metadata_clean.csv"))
    failed_csv_path = Path(args.failed_downloads_csv or data_cfg.get("failed_downloads_csv", "data/failed_downloads.csv"))

    limit = args.limit if args.limit is not None else ingest_cfg.get("limit")
    num_workers = args.workers or ingest_cfg.get("num_workers", 16)
    timeout = args.timeout or ingest_cfg.get("timeout", 20)
    retries = ingest_cfg.get("retries", 3)
    backoff_factor = ingest_cfg.get("backoff_factor", 1.5)
    rate_limit_interval = ingest_cfg.get("rate_limit_per_domain", 0.1)
    user_agent = ingest_cfg.get(
        "user_agent",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    )

    if not split_csv_path.exists():
        logger.error(f"Split CSV not found at {split_csv_path}")
        sys.exit(1)

    images_dir.mkdir(parents=True, exist_ok=True)
    clean_csv_path.parent.mkdir(parents=True, exist_ok=True)
    failed_csv_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading split CSV from {split_csv_path}")
    df = pd.read_csv(split_csv_path)

    # Print required dataset statistics
    print_dataset_statistics(df)

    if args.sync_existing:
        logger.info("Scanning and indexing all existing local images in images_dir...")
        clean_records: List[Dict[str, Any]] = []
        attempted_domains: List[str] = []
        for row in df.to_dict(orient="records"):
            filepath = str(row.get("filepath", "")).strip()
            target_path = images_dir / filepath
            if target_path.exists() and validate_image_file(target_path):
                domain = extract_domain(str(row.get("url", "")))
                attempted_domains.append(domain)
                clean_records.append({
                    "image_path": str(target_path),
                    "md5hash": str(row.get("md5hash", "")).strip(),
                    "label": row.get("label"),
                    "diag": row.get("diag"),
                    "nine_partition_label": row.get("nine_partition_label"),
                    "three_partition_label": row.get("three_partition_label"),
                    "fst": row.get("fst"),
                    "fitzpatrick": row.get("fitzpatrick"),
                    "partition": row.get("partition"),
                    "qc": row.get("qc"),
                    "domain": domain,
                })
        clean_df = pd.DataFrame(clean_records)
        clean_cols = [
            "image_path", "md5hash", "label", "diag", "nine_partition_label",
            "three_partition_label", "fst", "fitzpatrick", "partition", "qc"
        ]
        if not clean_df.empty:
            clean_df[clean_cols].to_csv(clean_csv_path, index=False)
            logger.info(f"Synchronized {len(clean_df)} verified local images into {clean_csv_path}")
        print_download_summary(clean_df, pd.DataFrame(), attempted_domains)
        return

    if limit is not None and limit > 0:
        logger.info(f"Limiting download run to first {limit} rows as requested (--limit {limit})")
        df_to_download = df.head(limit).copy()
    else:
        df_to_download = df.copy()

    logger.info(
        f"Starting download of {len(df_to_download)} images with {num_workers} threads, "
        f"timeout={timeout}s, retries={retries}"
    )

    rate_limiter = DomainRateLimiter(min_interval_seconds=rate_limit_interval)
    clean_records: List[Dict[str, Any]] = []
    failed_records: List[Dict[str, Any]] = []
    attempted_domains: List[str] = []

    rows = df_to_download.to_dict(orient="records")
    for r in rows:
        attempted_domains.append(extract_domain(str(r.get("url", ""))))

    session = requests.Session()

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_row = {
            executor.submit(
                download_single_image,
                row=row,
                images_dir=images_dir,
                session=session,
                rate_limiter=rate_limiter,
                user_agent=user_agent,
                timeout=timeout,
                max_retries=retries,
                backoff_factor=backoff_factor,
            ): row
            for row in rows
        }

        for future in tqdm(as_completed(future_to_row), total=len(future_to_row), desc="Downloading"):
            is_success, clean_rec, failed_rec = future.result()
            if is_success and clean_rec:
                clean_records.append(clean_rec)
            elif failed_rec:
                failed_records.append(failed_rec)

    # Build clean metadata DataFrame
    clean_cols = [
        "image_path",
        "md5hash",
        "label",
        "diag",
        "nine_partition_label",
        "three_partition_label",
        "fst",
        "fitzpatrick",
        "partition",
        "qc",
        "domain",
    ]
    clean_df = pd.DataFrame(clean_records)
    if not clean_df.empty:
        # Reorder columns
        ordered_cols = [c for c in clean_cols if c in clean_df.columns]
        clean_df = clean_df[ordered_cols]
        clean_df.to_csv(clean_csv_path, index=False)
        logger.info(f"Saved {len(clean_df)} clean records to {clean_csv_path}")
    else:
        # Write empty CSV with headers
        pd.DataFrame(columns=[c for c in clean_cols if c != "domain"]).to_csv(clean_csv_path, index=False)
        logger.info(f"No successful downloads. Wrote empty template to {clean_csv_path}")

    # Build failed downloads DataFrame
    failed_cols = ["md5hash", "url", "error", "domain"]
    failed_df = pd.DataFrame(failed_records)
    if not failed_df.empty:
        failed_df = failed_df[failed_cols]
        failed_df.to_csv(failed_csv_path, index=False)
        logger.info(f"Saved {len(failed_df)} failed records to {failed_csv_path}")
    else:
        pd.DataFrame(columns=failed_cols).to_csv(failed_csv_path, index=False)
        logger.info(f"No failed downloads. Wrote empty template to {failed_csv_path}")

    # Print summary statistics
    print_download_summary(clean_df, failed_df, attempted_domains)


if __name__ == "__main__":
    main()
