"""
Vishwas Training Pipeline: train.py

End-to-end training, validation, and testing supporting:
- Datasets: DermaMNIST (28x28 / 224x224) and Fitzpatrick17k-C
- Architectures: ResNet18 and EfficientNet-B0 with custom heads
- Mitigations: none | class_weighted | balanced_sampler | group_weighted
- Device: Apple Silicon MPS / CPU
- Artifacts: predictions.csv, confusion_matrix.png, metrics.md, experiments.csv
- Automated Fairness Evaluation: Mode (a) with skin-tones or Mode (b) without skin-tones
"""

from __future__ import annotations

import argparse
import copy
import logging
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
from torch.utils.data import DataLoader, WeightedRandomSampler
import yaml

# Support both workspace root and package execution
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent.parent
VISHWAS_ROOT = Path(__file__).resolve().parent.parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))
if str(VISHWAS_ROOT) not in sys.path:
    sys.path.insert(0, str(VISHWAS_ROOT))

try:
    from vishwas.src.models.classifier import build_model
    from vishwas.src.preprocess.dataset import build_dataloaders, GROUP_TO_IDX
    from vishwas.src.fairness.evaluate import evaluate_run
except ImportError:
    from src.models.classifier import build_model
    from src.preprocess.dataset import build_dataloaders, GROUP_TO_IDX
    from src.fairness.evaluate import evaluate_run

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("vishwas.train")


def set_seed(seed: int = 42) -> None:
    """Sets fixed seeds for deterministic reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(requested: str = "mps") -> torch.device:
    """Selects Apple Silicon MPS if available, otherwise falls back to CPU."""
    if requested == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    elif requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def compute_class_weights(dataset: Any, num_classes: int) -> torch.Tensor:
    """Computes inverse class frequency weights."""
    if hasattr(dataset, "labels"):
        # DermaMNIST dataset
        lbls = dataset.labels
        unique, counts = np.unique(lbls, return_counts=True)
        count_dict = dict(zip(unique, counts))
        raw_counts = np.array([count_dict.get(i, 1) for i in range(num_classes)], dtype=np.float32)
    elif hasattr(dataset, "df"):
        # Fitzpatrick dataset
        target_col = dataset.target_col
        count_dict = dataset.df[target_col].value_counts().to_dict()
        raw_counts = np.array([count_dict.get(dataset.idx_to_label.get(i), 1) for i in range(num_classes)], dtype=np.float32)
    else:
        raw_counts = np.ones(num_classes, dtype=np.float32)

    raw_counts = np.maximum(raw_counts, 1.0)
    weights = np.sum(raw_counts) / (num_classes * raw_counts)
    return torch.tensor(weights, dtype=torch.float32)


def compute_group_weights(dataset: Any) -> Optional[Dict[int, float]]:
    """Computes inverse group weights when skin tone groups exist."""
    if not hasattr(dataset, "df") or "ita_group" not in dataset.df.columns:
        return None
    counts = {0: 1.0, 1: 1.0, 2: 1.0}
    for _, row in dataset.df.iterrows():
        grp = str(row.get("ita_group", row.get("fitzpatrick_group", "medium"))).lower()
        idx = GROUP_TO_IDX.get(grp, 1)
        counts[idx] = counts.get(idx, 0.0) + 1.0
    tot = sum(counts.values())
    return {idx: tot / (3.0 * count) for idx, count in counts.items()}


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    mitigation: str,
    group_weights: Optional[Dict[int, float]] = None,
) -> Tuple[float, float, float]:
    """Executes one training epoch."""
    model.train()
    running_loss = 0.0
    all_preds, all_targets = [], []

    for batch in dataloader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        group_one_hot = batch["group_one_hot"].to(device)
        ita_val = batch["ita_val"].to(device)

        optimizer.zero_grad()
        logits = model(images, group_one_hot=group_one_hot, ita_val=ita_val)

        if mitigation == "group_weighted" and group_weights is not None:
            group_idx = batch["group_idx"].to(device)
            unweighted_loss = nn.functional.cross_entropy(logits, labels, reduction="none")
            batch_weights = torch.tensor([group_weights.get(int(g.item()), 1.0) for g in group_idx], device=device)
            loss = torch.mean(unweighted_loss * batch_weights)
        else:
            loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(labels.detach().cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = accuracy_score(all_targets, all_preds)
    epoch_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    return epoch_loss, epoch_acc, epoch_f1


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, float]:
    """Evaluates model performance on validation set."""
    model.eval()
    running_loss = 0.0
    all_preds, all_targets = [], []

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            group_one_hot = batch["group_one_hot"].to(device)
            ita_val = batch["ita_val"].to(device)

            logits = model(images, group_one_hot=group_one_hot, ita_val=ita_val)
            loss = criterion(logits, labels)

            running_loss += loss.item() * images.size(0)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(labels.cpu().numpy())

    eval_loss = running_loss / len(dataloader.dataset)
    eval_acc = accuracy_score(all_targets, all_preds)
    eval_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    return eval_loss, eval_acc, eval_f1


def save_confusion_matrix_plot(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: List[str],
    save_path: Path,
) -> None:
    """Plots and saves confusion matrix."""
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))
    plt.figure(figsize=(9, 7))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
    )
    plt.title("Test Set Confusion Matrix", fontsize=12, pad=10)
    plt.xlabel("Predicted Diagnostic Class", fontsize=10)
    plt.ylabel("True Diagnostic Class", fontsize=10)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def save_metrics_markdown(
    accuracy: float,
    macro_f1: float,
    per_class_recalls: Dict[str, float],
    run_dir: Path,
    dataset_name: str,
) -> None:
    """Saves metrics.md with accuracy, macro-F1 and per-class recall."""
    lines = [
        f"# Model Evaluation Metrics: `{run_dir.name}`\n",
        f"- **Dataset**: `{dataset_name}`",
        f"- **Overall Test Accuracy**: `{accuracy * 100:.2f}%`",
        f"- **Macro-Averaged F1**: `{macro_f1:.4f}`\n",
        "## Per-Class Recall (Sensitivity)\n",
        "| Class ID | Diagnostic Class Name | Recall (Sensitivity) |",
        "| :--- | :--- | :--- |",
    ]
    for i, (name, rec) in enumerate(per_class_recalls.items()):
        lines.append(f"| {i} | **{name}** | `{rec * 100:.2f}%` |")

    lines.append("\n## Visual Artifacts\n")
    lines.append("- [Confusion Matrix Plot](confusion_matrix.png)")

    with open(run_dir / "metrics.md", "w") as f:
        f.write("\n".join(lines))


def run_training(
    config_path: str = "vishwas/configs/config.yaml",
    mitigation_override: Optional[str] = None,
    run_name_override: Optional[str] = None,
    image_dir_override: Optional[str] = None,
    dataset_override: Optional[str] = None,
) -> str:
    """Main Vishwas model training workflow."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    seed = cfg.get("seed", 42)
    set_seed(seed)
    device = get_device(cfg.get("device", "mps"))
    logger.info(f"Using compute device: {device} (Apple Silicon MPS: {device.type == 'mps'})")

    # Build dataloaders
    dataloaders, dataset_info = build_dataloaders(
        config_path=config_path,
        image_dir_override=image_dir_override,
        dataset_override=dataset_override,
    )
    dataset_name = dataset_info["dataset_name"]
    has_skin_tone = dataset_info.get("has_skin_tone", False)
    num_classes = dataset_info["num_classes"]
    classes = dataset_info["classes"]
    idx_to_label = dataset_info["idx_to_label"]

    model_cfg = cfg.get("models", {})
    train_cfg = cfg.get("train", {})

    mitigation = mitigation_override or train_cfg.get("mitigation", "none")
    model_type = model_cfg.get("model_type", "image_only")
    backbone = model_cfg.get("backbone", "resnet18")
    epochs = train_cfg.get("epochs", 5)
    lr = float(train_cfg.get("lr", 0.0003))
    weight_decay = float(train_cfg.get("weight_decay", 0.01))
    patience = train_cfg.get("early_stopping_patience", 3)

    # Validate mitigation compatibility
    if mitigation in ("balanced_sampler", "group_weighted") and not has_skin_tone:
        logger.warning(
            f"Mitigation '{mitigation}' is only applicable when demographic skin-tone groups exist. "
            f"Dataset '{dataset_name}' lacks skin-tone groups. Falling back to mitigation: 'none'."
        )
        mitigation = "none"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = run_name_override or f"{dataset_name}_{backbone}_{mitigation}_{timestamp}"
    run_dir = Path(cfg["reports"].get("output_dir", "vishwas/reports")) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        f"Starting training run: {run_name} | Dataset: {dataset_name} | "
        f"Backbone: {backbone} | Mitigation: {mitigation} | Classes: {num_classes}"
    )

    # Build model
    model = build_model(
        model_type=model_type,
        backbone=backbone,
        num_classes=num_classes,
        pretrained=model_cfg.get("pretrained", True),
        dropout=float(model_cfg.get("dropout", 0.2)),
    ).to(device)

    # Class-weighted loss mitigation
    if mitigation == "class_weighted":
        weights = compute_class_weights(dataloaders["train"].dataset, num_classes).to(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
    else:
        criterion = nn.CrossEntropyLoss()

    group_weights = compute_group_weights(dataloaders["train"].dataset) if mitigation == "group_weighted" else None

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_f1 = -1.0
    best_model_wts = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0
    history: List[Dict[str, Any]] = []

    print("\n" + "=" * 65)
    print(f"               TRAINING PROGRESS: {run_name}")
    print("=" * 65)

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc, tr_f1 = train_one_epoch(
            model=model,
            dataloader=dataloaders["train"],
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            mitigation=mitigation,
            group_weights=group_weights,
        )
        scheduler.step()

        val_loss, val_acc, val_f1 = evaluate(model, dataloaders["val"], criterion, device)
        elapsed = time.time() - t0

        history.append({
            "epoch": epoch,
            "train_loss": tr_loss,
            "train_acc": tr_acc,
            "train_f1": tr_f1,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_f1": val_f1,
            "runtime_sec": round(elapsed, 2),
        })

        print(
            f"Epoch {epoch:02d}/{epochs:02d} [{elapsed:.1f}s] | "
            f"Train Acc: {tr_acc * 100:.1f}% F1: {tr_f1:.4f} | Val Acc: {val_acc * 100:.1f}% F1: {val_f1:.4f}"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_wts = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
            torch.save(best_model_wts, run_dir / "best_checkpoint.pt")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logger.info(f"Early stopping triggered at epoch {epoch}")
                break

    print("=" * 65 + "\n")

    # Save training curves
    pd.DataFrame(history).to_csv(run_dir / "training_curves.csv", index=False)

    # Load best checkpoint for testing
    model.load_state_dict(best_model_wts)
    model.eval()

    logger.info("Evaluating best model on test set...")
    test_records = []
    all_probs = []

    with torch.no_grad():
        for batch in dataloaders["test"]:
            images = batch["image"].to(device)
            group_one_hot = batch["group_one_hot"].to(device)
            ita_val = batch["ita_val"].to(device)

            logits = model(images, group_one_hot=group_one_hot, ita_val=ita_val)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)
            max_probs = np.max(probs, axis=1)

            for i in range(len(preds)):
                rec = {
                    "image_id": batch["image_id"][i],
                    "y_true": int(batch["label"][i].item()),
                    "y_true_name": batch["label_name"][i],
                    "y_pred": int(preds[i]),
                    "y_pred_name": idx_to_label.get(int(preds[i]), "unknown"),
                    "y_prob": float(max_probs[i]),
                }
                # Optional skin-tone columns
                if has_skin_tone:
                    rec["skin_tone_group"] = batch["skin_tone_group"][i]
                    rec["fitzpatrick"] = int(batch["fitzpatrick"][i])
                    rec["ita_val"] = float(batch["ita_val"][i].item())
                test_records.append(rec)
            all_probs.append(probs)

    pred_df = pd.DataFrame(test_records)
    pred_df.to_csv(run_dir / "predictions.csv", index=False)

    concatenated_probs = np.concatenate(all_probs, axis=0)
    np.save(run_dir / "probabilities.npy", concatenated_probs)

    test_acc = accuracy_score(pred_df["y_true"], pred_df["y_pred"])
    test_f1 = f1_score(pred_df["y_true"], pred_df["y_pred"], average="macro", zero_division=0)

    # Per-class recall
    per_class_recalls = {}
    recalls_arr = recall_score(pred_df["y_true"], pred_df["y_pred"], labels=list(range(num_classes)), average=None, zero_division=0)
    for c_idx, rec_val in enumerate(recalls_arr):
        per_class_recalls[idx_to_label[c_idx]] = float(rec_val)

    # Save confusion matrix plot and metrics.md
    save_confusion_matrix_plot(
        y_true=pred_df["y_true"].values,
        y_pred=pred_df["y_pred"].values,
        classes=classes,
        save_path=run_dir / "confusion_matrix.png",
    )

    save_metrics_markdown(
        accuracy=test_acc,
        macro_f1=test_f1,
        per_class_recalls=per_class_recalls,
        run_dir=run_dir,
        dataset_name=dataset_name,
    )

    # Run fairness audit (Mode A or Mode B)
    evaluate_run(run_dir)

    # Append to reports/experiments.csv
    exp_log_path = Path(cfg["reports"].get("experiments_log", "vishwas/reports/experiments.csv"))
    exp_log_path.parent.mkdir(parents=True, exist_ok=True)
    avg_epoch_time = float(np.mean([h["runtime_sec"] for h in history]))
    exp_entry = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "run_name": run_name,
        "dataset": dataset_name,
        "backbone": backbone,
        "mitigation": mitigation,
        "augmentation": dataset_info["augmentation"],
        "num_classes": num_classes,
        "best_val_f1": round(best_val_f1, 4),
        "test_acc": round(test_acc, 4),
        "test_f1": round(test_f1, 4),
        "epochs": len(history),
        "avg_epoch_time_sec": round(avg_epoch_time, 2),
        "seed": seed,
    }
    exp_df = pd.DataFrame([exp_entry])
    if exp_log_path.exists():
        exp_df.to_csv(exp_log_path, mode="a", header=False, index=False)
    else:
        exp_df.to_csv(exp_log_path, index=False)

    print("\n" + "=" * 65)
    print("                 FINAL EVALUATION SUMMARY")
    print("=" * 65)
    print(f"Run name          : {run_name}")
    print(f"Test Accuracy     : {test_acc * 100:.2f}%")
    print(f"Test Macro-F1     : {test_f1:.4f}")
    print(f"Avg Runtime/epoch : {avg_epoch_time:.2f} s")
    print("\nPer-Class Recall:")
    for c_name, c_rec in per_class_recalls.items():
        print(f"  - {c_name:<45}: {c_rec * 100:.2f}%")
    print("=" * 65 + "\n")
    print(f"All artifacts saved to: {run_dir.resolve()}")
    return run_name


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Vishwas classifier")
    parser.add_argument("--config", type=str, default="vishwas/configs/config.yaml")
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--mitigation", type=str, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--image-dir", type=str, default=None)
    args = parser.parse_args()

    run_training(
        config_path=args.config,
        mitigation_override=args.mitigation,
        run_name_override=args.run_name,
        image_dir_override=args.image_dir,
        dataset_override=args.dataset,
    )
