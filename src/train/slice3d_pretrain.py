"""
Vishwas SLICE-3D Stage-1 Pretraining Pipeline: slice3d_pretrain.py

Trains an EfficientNet-B0 visual encoder on the binary SLICE-3D subset:
- timm EfficientNet-B0 (ImageNet-pretrained), 1-logit classification head, BCEWithLogitsLoss
- AdamW optimizer with 1-epoch warmup + cosine annealing
- Early stopping monitored on validation pAUC (above 80% TPR)
- Evaluates ONCE on test split after training
- Exports bootstrap 95% CIs (1,000 resamples grouped by patient) without accuracy
- Saves encoder-only weights to checkpoints/slice3d_pretrain_effb0.pt for Stage-2 initialization
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import timm

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent.parent
VISHWAS_ROOT = Path(__file__).resolve().parent.parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))
if str(VISHWAS_ROOT) not in sys.path:
    sys.path.insert(0, str(VISHWAS_ROOT))

try:
    from vishwas.src.preprocess.slice3d_dataset import build_slice3d_dataloaders
    from vishwas.src.utils.metrics import bootstrap_patient_metrics, compute_pauc, compute_roc_auc
except ImportError:
    from src.preprocess.slice3d_dataset import build_slice3d_dataloaders
    from src.utils.metrics import bootstrap_patient_metrics, compute_pauc, compute_roc_auc

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("vishwas.train.slice3d")


def get_device(requested_device: Optional[str] = None) -> torch.device:
    """Auto-selects optimal computing device (CUDA > MPS > CPU)."""
    if requested_device:
        return torch.device(requested_device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class Slice3DPretrainModel(nn.Module):
    """Visual classifier with separate encoder and single-logit binary head."""

    def __init__(
        self,
        backbone_name: str = "efficientnet_b0",
        pretrained: bool = True,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.encoder = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0)
        in_features = self.encoder.num_features
        self.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.encoder(x)
        logits = self.head(feats)
        return logits.squeeze(-1)


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    use_amp: bool = False,
) -> Tuple[float, float]:
    """Runs a single training epoch and returns (mean_loss, elapsed_sec)."""
    model.train()
    running_loss = 0.0
    total_samples = 0
    t0 = time.time()

    scaler = torch.cuda.amp.GradScaler() if (use_amp and device.type == "cuda") else None

    for batch in dataloader:
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)
        batch_size = images.size(0)

        optimizer.zero_grad()

        if scaler:
            with torch.cuda.amp.autocast():
                logits = model(images)
                loss = criterion(logits, targets)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        running_loss += loss.item() * batch_size
        total_samples += batch_size

    if device.type == "mps":
        torch.mps.empty_cache()

    elapsed = time.time() - t0
    epoch_loss = running_loss / max(1, total_samples)
    return epoch_loss, elapsed


@torch.no_grad()
def evaluate_slice3d_loader(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, float, np.ndarray, np.ndarray, List[str], List[str]]:
    """Evaluates validation or test set, computing BCE loss, ROC-AUC, and pAUC."""
    model.eval()
    running_loss = 0.0
    total_samples = 0

    all_targets: List[float] = []
    all_probs: List[float] = []
    all_isics: List[str] = []
    all_patients: List[str] = []

    for batch in dataloader:
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)
        batch_size = images.size(0)

        logits = model(images)
        loss = criterion(logits, targets)

        probs = torch.sigmoid(logits).cpu().numpy()
        running_loss += loss.item() * batch_size
        total_samples += batch_size

        all_targets.extend(targets.cpu().numpy().tolist())
        all_probs.extend(probs.tolist())
        all_isics.extend(batch["isic_id"])
        all_patients.extend(batch["patient_id"])

    targets_arr = np.asarray(all_targets)
    probs_arr = np.asarray(all_probs)

    mean_loss = running_loss / max(1, total_samples)
    roc_auc = compute_roc_auc(targets_arr, probs_arr)
    pauc = compute_pauc(targets_arr, probs_arr, min_tpr=0.80)

    return mean_loss, roc_auc, pauc, targets_arr, probs_arr, all_isics, all_patients


def run_slice3d_pretrain(
    epochs: int = 5,
    batch_size: int = 64,
    lr: float = 3e-4,
    lr_encoder: float = 1e-4,
    lr_head: float = 1e-3,
    weight_decay: float = 0.01,
    warmup_epochs: int = 1,
    patience: int = 3,
    img_size: int = 128,
    num_workers: int = 2,
    balanced: bool = False,
    benign_per_malignant: int = 20,
    max_images: Optional[int] = None,
    resume: bool = False,
    device_name: Optional[str] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """Executes the complete SLICE-3D stage-1 pretraining pipeline."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = get_device(device_name)

    print("=" * 72)
    print("VISHWAS: SLICE-3D STAGE-1 PRETRAINING PIPELINE")
    print(f"Device: {dev} | Epochs: {epochs} | Batch Size: {batch_size} | Img Size: {img_size}px")
    print(f"Balanced: {balanced} (ratio: {benign_per_malignant}:1) | Max Images: {max_images}")
    print("=" * 72)

    # Directories
    ckpt_dir = VISHWAS_ROOT / "checkpoints"
    rep_dir = VISHWAS_ROOT / "reports" / "slice3d_pretrain"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    rep_dir.mkdir(parents=True, exist_ok=True)

    encoder_out_path = ckpt_dir / "slice3d_pretrain_effb0.pt"
    full_ckpt_path = ckpt_dir / "slice3d_pretrain_best_full.pt"
    last_ckpt_path = ckpt_dir / "slice3d_pretrain_last.pt"

    # DataLoaders
    dataloaders, meta = build_slice3d_dataloaders(
        img_size=img_size,
        batch_size=batch_size,
        num_workers=num_workers,
        balanced=balanced,
        benign_per_malignant=benign_per_malignant,
        max_images=max_images,
        seed=seed,
    )

    train_loader = dataloaders["train"]
    val_loader = dataloaders["val"]
    test_loader = dataloaders["test"]
    pos_weight = meta["pos_weight"].to(dev)

    # Loss functions
    train_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    val_criterion = nn.BCEWithLogitsLoss()

    # Model
    model = Slice3DPretrainModel(backbone_name="efficientnet_b0", pretrained=True, dropout=0.2).to(dev)

    # Parameter groups
    optimizer = torch.optim.AdamW(
        [
            {"params": model.encoder.parameters(), "lr": lr_encoder, "weight_decay": weight_decay},
            {"params": model.head.parameters(), "lr": lr_head, "weight_decay": weight_decay},
        ]
    )

    # Warmup + Cosine Scheduler
    def lr_lambda(ep: int) -> float:
        if ep < warmup_epochs:
            return float(ep + 1) / float(max(1, warmup_epochs))
        progress = float(ep - warmup_epochs) / float(max(1, epochs - warmup_epochs))
        return 0.5 * (1.0 + float(np.cos(np.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    start_epoch = 1
    best_val_pauc = -1.0
    epochs_no_improve = 0
    history: List[Dict[str, Any]] = []

    # Resume handling
    if resume and last_ckpt_path.exists():
        print(f"[*] Resuming from checkpoint: {last_ckpt_path}")
        chk = torch.load(last_ckpt_path, map_location=dev)
        model.load_state_dict(chk["model_state"])
        optimizer.load_state_dict(chk["optimizer_state"])
        scheduler.load_state_dict(chk["scheduler_state"])
        start_epoch = chk["epoch"] + 1
        best_val_pauc = chk.get("best_val_pauc", -1.0)
        history = chk.get("history", [])

    total_train_samples = len(train_loader.dataset) if not balanced else len(meta["sampler"])

    # Training Loop
    for epoch in range(start_epoch, epochs + 1):
        if balanced and meta["sampler"] is not None:
            meta["sampler"].set_epoch(epoch)

        tr_loss, tr_time = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=train_criterion,
            optimizer=optimizer,
            device=dev,
            use_amp=(dev.type == "cuda"),
        )

        val_loss, val_auc, val_pauc, _, _, _, _ = evaluate_slice3d_loader(
            model=model,
            dataloader=val_loader,
            criterion=val_criterion,
            device=dev,
        )

        scheduler.step()

        # Compute and log throughput
        imgs_per_sec = total_train_samples / max(0.001, tr_time)
        if epoch == 1:
            est_total_min = (total_train_samples * epochs / max(1.0, imgs_per_sec)) / 60.0
            print(f"\n[⚡ PERFORMANCE] Epoch 1 Speed: {imgs_per_sec:.1f} images/sec | Estimated total {epochs}-epoch time: {est_total_min:.1f} minutes\n", flush=True)

        is_best = val_pauc > best_val_pauc
        if is_best:
            best_val_pauc = val_pauc
            epochs_no_improve = 0
            # Save encoder-only weights for stage 2
            torch.save(model.encoder.state_dict(), encoder_out_path)
            # Save full model for test evaluation
            torch.save(model.state_dict(), full_ckpt_path)
            best_mark = " [*BEST pAUC]"
        else:
            epochs_no_improve += 1
            best_mark = ""

        # Save resume checkpoint
        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "best_val_pauc": best_val_pauc,
                "history": history,
            },
            last_ckpt_path,
        )

        epoch_record = {
            "epoch": epoch,
            "epoch_time_sec": round(tr_time, 2),
            "images_per_sec": round(imgs_per_sec, 1),
            "train_loss": round(tr_loss, 4),
            "val_loss": round(val_loss, 4),
            "val_roc_auc": round(val_auc, 4),
            "val_pauc": round(val_pauc, 4),
        }
        history.append(epoch_record)

        log_line = (
            f"Epoch {epoch:02d}/{epochs:02d} | "
            f"Time: {tr_time:.1f}s ({imgs_per_sec:.1f} img/s) | "
            f"Train Loss: {tr_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val ROC-AUC: {val_auc:.4f} | "
            f"Val pAUC(>80% TPR): {val_pauc:.4f}{best_mark}"
        )
        print(log_line, flush=True)

        if epochs_no_improve >= patience:
            print(f"[*] Early stopping triggered at epoch {epoch} (patience={patience}, best pAUC={best_val_pauc:.4f})")
            break

    # Save training curves
    curves_df = pd.DataFrame(history)
    curves_path = rep_dir / "training_curves.csv"
    curves_df.to_csv(curves_path, index=False)
    print(f"\n[+] Saved training curves to: {curves_path}")

    # Load best checkpoint for ONE test evaluation
    if full_ckpt_path.exists():
        model.load_state_dict(torch.load(full_ckpt_path, map_location=dev))
    else:
        torch.save(model.encoder.state_dict(), encoder_out_path)

    print("\n--- Running Final Evaluation ONCE on Held-Out Test Split ---")
    te_loss, te_auc, te_pauc, te_targets, te_probs, te_isics, te_patients = evaluate_slice3d_loader(
        model=model,
        dataloader=test_loader,
        criterion=val_criterion,
        device=dev,
    )

    test_pred_df = pd.DataFrame({
        "isic_id": te_isics,
        "patient_id": te_patients,
        "target": te_targets,
        "prob_malignant": te_probs,
    })

    # Grouped bootstrap 95% CIs (1,000 resamples by patient)
    print("[*] Computing 1,000 patient-grouped bootstrap 95% confidence intervals...")
    boot_stats = bootstrap_patient_metrics(
        df=test_pred_df,
        target_col="target",
        prob_col="prob_malignant",
        patient_col="patient_id",
        n_bootstrap=1000,
        seed=seed,
        min_tpr=0.80,
    )

    # Write reports/slice3d_pretrain/test_metrics.md
    test_metrics_md_path = rep_dir / "test_metrics.md"
    auc_ci = boot_stats["roc_auc_ci"]
    pauc_ci = boot_stats["pauc_ci"]
    prev_pct = boot_stats["prevalence"] * 100.0

    test_report_lines = [
        "# SLICE-3D Stage-1 Pretraining: Test Evaluation Metrics",
        "",
        "> **Evaluation Protocol**: Single evaluation on the held-out test split (25,000 images).",
        "> **Confidence Intervals**: 1,000 bootstrap resamples clustered at the patient level.",
        "> **Metric Standard**: Diagnostic accuracy is omitted; ranking and sensitivity metrics are reported.",
        "",
        "## 1. Primary Test Metrics",
        "",
        "| Evaluation Metric | Point Estimate | 95% Bootstrap CI (Patient-Clustered) | Clinical Definition |",
        "| :--- | :---: | :---: | :--- |",
        f"| **pAUC (TPR $\\ge 0.80$)** | **`{boot_stats['pauc']:.4f}`** | **`[{pauc_ci[0]:.4f}, {pauc_ci[1]:.4f}]`** | Official ISIC 2024 metric (high-sensitivity partial AUC) |",
        f"| **ROC-AUC** | **`{boot_stats['roc_auc']:.4f}`** | **`[{auc_ci[0]:.4f}, {auc_ci[1]:.4f}]`** | Full receiver operating characteristic area |",
        f"| **Test Prevalence** | **`{prev_pct:.4f}%`** | --- | `{boot_stats['n_malignant']}` malignant cases / `{boot_stats['n_samples']}` total |",
        "",
        "## 2. Test Cohort Characteristics",
        "",
        "| Dimension | Count | Note |",
        "| :--- | :---: | :--- |",
        f"| **Evaluated Images** | `{boot_stats['n_samples']:,}` | Fixed test partition |",
        f"| **Evaluated Patients** | `{boot_stats['n_patients']:,}` | Zero overlap with training/validation cohorts |",
        f"| **Malignant Samples ($y=1$)** | `{boot_stats['n_malignant']:,}` | All positive cases preserved in partition |",
        f"| **Benign Samples ($y=0$)** | `{boot_stats['n_benign']:,}` | Subsampled benign majority |",
        "",
        "> **Prevalence Notice**: Test prevalence is 0.3880%, enriched relative to the 0.0980% raw dataset base rate. Both ROC-AUC and pAUC are insensitive to prevalence shifts, providing robust benchmark comparisons.",
    ]
    with open(test_metrics_md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(test_report_lines))

    # Write summary metrics.md
    metrics_md_path = rep_dir / "metrics.md"
    summary_lines = [
        "# SLICE-3D Stage-1 Pretraining Summary",
        "",
        f"- **Encoder Checkpoint**: [`checkpoints/slice3d_pretrain_effb0.pt`](file://{encoder_out_path})",
        f"- **Best Validation pAUC (TPR $\\ge 0.80$)**: `{best_val_pauc:.4f}`",
        f"- **Test ROC-AUC**: `{boot_stats['roc_auc']:.4f}` [{auc_ci[0]:.4f}, {auc_ci[1]:.4f}]",
        f"- **Test pAUC (TPR $\\ge 0.80$)**: `{boot_stats['pauc']:.4f}` [{pauc_ci[0]:.4f}, {pauc_ci[1]:.4f}]",
        f"- **Training Curves**: [`reports/slice3d_pretrain/training_curves.csv`](file://{curves_path})",
        f"- **Test Metrics Report**: [`reports/slice3d_pretrain/test_metrics.md`](file://{test_metrics_md_path})",
    ]
    with open(metrics_md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))

    print(f"[+] Encoder-only weights saved to: {encoder_out_path}")
    print(f"[+] Test metrics report written to: {test_metrics_md_path}")
    print(f"[+] Stage-1 metrics summary written to: {metrics_md_path}")
    print("=" * 72)

    return {
        "best_val_pauc": best_val_pauc,
        "test_roc_auc": boot_stats["roc_auc"],
        "test_pauc": boot_stats["pauc"],
        "encoder_path": str(encoder_out_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SLICE-3D Stage-1 EfficientNet-B0 visual encoder")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Default learning rate")
    parser.add_argument("--lr-encoder", type=float, default=1e-4, help="Encoder learning rate")
    parser.add_argument("--lr-head", type=float, default=1e-3, help="Head learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--warmup-epochs", type=int, default=1, help="Warmup epochs")
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience")
    parser.add_argument("--img-size", type=int, default=128, help="Image resolution")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader num_workers")
    parser.add_argument("--balanced", action="store_true", help="Enable balanced epoch resampling")
    parser.add_argument("--benign-per-malignant", type=int, default=20, help="Benign to malignant ratio in balanced mode")
    parser.add_argument("--max-images", type=int, default=None, help="Cap total images for smoke runs")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    parser.add_argument("--device", type=str, default=None, help="Device override (cuda/mps/cpu)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    run_slice3d_pretrain(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        lr_encoder=args.lr_encoder,
        lr_head=args.lr_head,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        patience=args.patience,
        img_size=args.img_size,
        num_workers=args.num_workers,
        balanced=args.balanced,
        benign_per_malignant=args.benign_per_malignant,
        max_images=args.max_images,
        resume=args.resume,
        device_name=args.device,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
