"""
Vishwas MILK10k Training Pipeline: milk10k_train.py

Implements 5-fold cross-validation and single-fold training:
- Architectures: dermoscopic_only, clinical_only, two_image_fusion, two_image_tabular_fusion
- Separate encoders via timm EfficientNet-B0
- Mitigations: none | class_weighted | balanced_sampler | group_weighted (by tone_group)
- 10% internal validation slice (grouped by lesion, stratified by diagnosis)
- Early stopping on validation headline macro-F1
- Cosine LR scheduling, AdamW, gradient clipping, Apple MPS/CPU acceleration
- Checkpointing best model weights to checkpoints/
- Predictions logging to reports/<run_name>/predictions.csv
- Automated fairness evaluation via evaluate_milk10k_predictions
- Appends to reports/experiments.csv
"""

from __future__ import annotations

import logging
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

logger = logging.getLogger("vishwas.train.milk10k")

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent.parent
VISHWAS_ROOT = Path(__file__).resolve().parent.parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))
if str(VISHWAS_ROOT) not in sys.path:
    sys.path.insert(0, str(VISHWAS_ROOT))

try:
    from vishwas.src.models.classifier import build_model
    from vishwas.src.preprocess.milk10k_dataset import (
        HEADLINE_CLASSES,
        MILK10K_CLASSES,
        MILK10K_CLASS_TO_IDX,
        MILK10K_IDX_TO_CLASS,
        load_milk10k_splits,
    )
    from vishwas.src.fairness.milk10k_eval import evaluate_milk10k_predictions
except ImportError:
    from src.models.classifier import build_model
    from src.preprocess.milk10k_dataset import (
        HEADLINE_CLASSES,
        MILK10K_CLASSES,
        MILK10K_CLASS_TO_IDX,
        MILK10K_IDX_TO_CLASS,
        load_milk10k_splits,
    )
    from src.fairness.milk10k_eval import evaluate_milk10k_predictions


def set_seed(seed: int = 42) -> None:
    """Sets random seeds across random, numpy, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Selects Apple Silicon MPS if available, otherwise CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def compute_milk10k_class_weights(dataset: Any, num_classes: int = 11, cap: float = 10.0) -> torch.Tensor:
    """Computes inverse class frequency weights on training set, capped at cap."""
    if hasattr(dataset, "lesions") and "diagnosis" in dataset.lesions.columns:
        labels = [MILK10K_CLASS_TO_IDX[d] for d in dataset.lesions["diagnosis"]]
    elif hasattr(dataset, "indices") and hasattr(dataset, "dataset") and hasattr(dataset.dataset, "lesions"):
        labels = [MILK10K_CLASS_TO_IDX[dataset.dataset.lesions.iloc[i]["diagnosis"]] for i in dataset.indices]
    else:
        labels = [b["label"].item() if isinstance(b["label"], torch.Tensor) else int(b["label"]) for b in dataset]
    counts = np.bincount(labels, minlength=num_classes)
    counts = np.maximum(counts, 1.0)
    weights = len(labels) / (num_classes * counts)
    if cap is not None and cap > 0:
        weights = np.minimum(weights, float(cap))
    return torch.tensor(weights, dtype=torch.float32)


def get_parameter_groups(
    model: nn.Module,
    lr_encoder: float = 1e-4,
    lr_head: float = 1e-3,
    weight_decay: float = 0.01,
) -> List[Dict[str, Any]]:
    """Partitions model parameters into pretrained encoder vs classification/fusion head."""
    encoder_params: List[nn.Parameter] = []
    head_params: List[nn.Parameter] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(enc in name for enc in ["encoder", "clinical_encoder", "dermoscopic_encoder"]):
            encoder_params.append(param)
        else:
            head_params.append(param)

    groups = []
    if encoder_params:
        groups.append({
            "params": encoder_params,
            "lr": lr_encoder,
            "weight_decay": weight_decay,
        })
    if head_params:
        groups.append({
            "params": head_params,
            "lr": lr_head,
            "weight_decay": weight_decay,
        })
    return groups


def build_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_epochs: int = 1,
    total_epochs: int = 12,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Constructs 1-epoch linear warmup followed by cosine annealing decay."""
    def lr_lambda(ep: int) -> float:
        if ep < warmup_epochs:
            return float(ep + 1) / float(max(1, warmup_epochs))
        else:
            progress = float(ep - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
            return 0.5 * (1.0 + float(np.cos(np.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def compute_tone_group_weights(dataset: Any) -> Dict[str, float]:
    """Computes inverse frequency weights for tone groups (G1, G2, G3, unknown)."""
    if hasattr(dataset, "lesions") and "tone_group" in dataset.lesions.columns:
        groups = dataset.lesions["tone_group"].tolist()
    elif hasattr(dataset, "indices") and hasattr(dataset, "dataset") and hasattr(dataset.dataset, "lesions"):
        groups = [dataset.dataset.lesions.iloc[i]["tone_group"] for i in dataset.indices]
    else:
        groups = [b["tone_group"] for b in dataset]
    unique_grps, counts = np.unique(groups, return_counts=True)
    count_dict = dict(zip(unique_grps, counts))
    n_tot = len(groups)
    n_grps = len(unique_grps)
    weights = {g: float(n_tot / (n_grps * count_dict[g])) for g in unique_grps}
    return weights


def forward_model_batch(
    model: nn.Module,
    model_name: str,
    batch: Dict[str, Any],
    device: torch.device,
) -> torch.Tensor:
    """Passes inputs to the appropriate model variant."""
    c_img = batch["clinical_img"].to(device)
    d_img = batch["dermoscopic_img"].to(device)
    tab = batch["tabular"].to(device)

    m_name = model_name.lower()
    if m_name in ["dermoscopic_only", "dermoscopic", "dermo_only"]:
        return model(dermoscopic_img=d_img)
    elif m_name in ["clinical_only", "clinical", "clin_only"]:
        return model(clinical_img=c_img)
    elif m_name in ["two_image_fusion", "image_fusion", "dual_image"]:
        return model(clinical_img=c_img, dermoscopic_img=d_img)
    else:  # two_image_tabular_fusion
        return model(clinical_img=c_img, dermoscopic_img=d_img, tabular=tab)


def train_one_epoch_milk10k(
    model: nn.Module,
    dataloader: DataLoader,
    model_name: str,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    mitigation: str,
    group_weights: Optional[Dict[str, float]] = None,
) -> Tuple[float, float, float]:
    """Executes a single training epoch."""
    model.train()
    running_loss = 0.0
    all_preds, all_targets = [], []

    use_autocast = device.type == "cuda"

    for batch in dataloader:
        labels = batch["label"].to(device)
        optimizer.zero_grad()

        if use_autocast:
            with torch.autocast(device_type=device.type, dtype=torch.float16):
                logits = forward_model_batch(model, model_name, batch, device)
                if mitigation == "group_weighted" and group_weights is not None:
                    unreduced = nn.functional.cross_entropy(logits, labels, reduction="none")
                    sample_w = torch.tensor(
                        [group_weights.get(g, 1.0) for g in batch["tone_group"]],
                        device=device,
                        dtype=torch.float32,
                    )
                    loss = torch.mean(unreduced * sample_w)
                else:
                    loss = criterion(logits, labels)
        else:
            logits = forward_model_batch(model, model_name, batch, device)
            if mitigation == "group_weighted" and group_weights is not None:
                unreduced = nn.functional.cross_entropy(logits, labels, reduction="none")
                sample_w = torch.tensor(
                    [group_weights.get(g, 1.0) for g in batch["tone_group"]],
                    device=device,
                    dtype=torch.float32,
                )
                loss = torch.mean(unreduced * sample_w)
            else:
                loss = criterion(logits, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item() * len(labels)
        preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(labels.detach().cpu().numpy())

    if device.type == "mps":
        torch.mps.empty_cache()

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = accuracy_score(all_targets, all_preds)
    headline_indices = [MILK10K_CLASS_TO_IDX[c] for c in HEADLINE_CLASSES]
    epoch_f1 = f1_score(all_targets, all_preds, labels=headline_indices, average="macro", zero_division=0)
    return epoch_loss, epoch_acc, epoch_f1


def evaluate_milk10k_loader(
    model: nn.Module,
    dataloader: DataLoader,
    model_name: str,
    criterion: nn.Module,
    device: torch.device,
    metric_indices: Optional[List[int]] = None,
) -> Tuple[float, float, float, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluates on validation or test loader."""
    model.eval()
    running_loss = 0.0
    all_preds, all_targets, all_probs = [], [], []

    use_autocast = device.type == "cuda"

    with torch.no_grad():
        for batch in dataloader:
            labels = batch["label"].to(device)

            if use_autocast:
                with torch.autocast(device_type=device.type, dtype=torch.float16):
                    logits = forward_model_batch(model, model_name, batch, device)
                    loss = criterion(logits, labels)
            else:
                logits = forward_model_batch(model, model_name, batch, device)
                loss = criterion(logits, labels)

            running_loss += loss.item() * len(labels)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)

            all_probs.append(probs)
            all_preds.extend(preds)
            all_targets.extend(labels.cpu().numpy())

    eval_loss = running_loss / len(dataloader.dataset)
    all_preds_arr = np.array(all_preds)
    all_targets_arr = np.array(all_targets)
    all_probs_arr = np.concatenate(all_probs, axis=0)

    eval_acc = float(accuracy_score(all_targets_arr, all_preds_arr))
    eval_metric_indices = metric_indices if metric_indices is not None else [MILK10K_CLASS_TO_IDX[c] for c in HEADLINE_CLASSES]
    eval_f1 = float(f1_score(all_targets_arr, all_preds_arr, labels=eval_metric_indices, average="macro", zero_division=0))

    if device.type == "mps":
        torch.mps.empty_cache()

    return eval_loss, eval_acc, eval_f1, all_preds_arr, all_targets_arr, all_probs_arr


def train_milk10k_fold(
    fold: int,
    model_name: str = "two_image_tabular_fusion",
    epochs: int = 12,
    img_size: int = 128,
    run_name: str = "smoke_fold0",
    mitigation: str = "class_weighted",
    batch_size: int = 32,
    lr_encoder: float = 1e-4,
    lr_head: float = 1e-3,
    lr: float = 3e-4,
    weight_decay: float = 0.01,
    label_smoothing: float = 0.05,
    warmup_epochs: int = 1,
    patience: int = 4,
    min_val_samples_per_class: int = 5,
    class_weight_cap: float = 10.0,
    fusion_type: str = "concat",
    augmentation: str = "standard",
    use_monet: bool = False,
    seed: int = 42,
    device: Optional[torch.device] = None,
    checkpoints_dir: Optional[Path] = None,
    logger_instance: Optional[logging.Logger] = None,
) -> Tuple[pd.DataFrame, float, List[Dict[str, Any]]]:
    """
    Executes training on 4 training folds, early stops on 10% validation slice,
    and returns held-out fold predictions DataFrame.
    """
    set_seed(seed)
    dev = device or get_device()
    ckpt_dir = checkpoints_dir or (VISHWAS_ROOT / "checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = ckpt_dir / f"best_{run_name}_fold{fold}.pt"

    header_str = f" TRAINING FOLD {fold} | Model: {model_name} | Mitigation: {mitigation} | Device: {dev}"
    print(f"\n{'='*70}\n{header_str}\n{'='*70}", flush=True)
    if logger_instance:
        logger_instance.info(f"{'='*70}\n{header_str}\n{'='*70}")

    # Build DataLoaders
    dataloaders, info = load_milk10k_splits(
        held_out_fold=fold,
        val_slice_ratio=0.10,
        img_size=img_size,
        augmentation=augmentation,
        batch_size=batch_size,
        num_workers=0,
        use_monet=use_monet,
        seed=seed,
    )

    train_ds = dataloaders["train"].dataset
    val_ds = dataloaders["val"].dataset
    num_classes = len(MILK10K_CLASSES)

    # Validation Early Stopping Metric: classes with >= min_val_samples_per_class
    val_counts = val_ds.lesions["diagnosis"].value_counts().to_dict()
    eligible_classes = [c for c in MILK10K_CLASSES if val_counts.get(c, 0) >= min_val_samples_per_class]
    eligible_indices = [MILK10K_CLASS_TO_IDX[c] for c in eligible_classes]

    val_metric_info = (
        f"Validation Early Stopping: monitoring val macro-F1 over {len(eligible_classes)} classes "
        f"with >= {min_val_samples_per_class} lesions in val slice: {eligible_classes}"
    )
    print(f"[*] {val_metric_info}", flush=True)
    if logger_instance:
        logger_instance.info(val_metric_info)

    # Configure Loss & Sampler according to mitigation
    group_weights = None
    if mitigation == "class_weighted":
        weights = compute_milk10k_class_weights(train_ds, num_classes=num_classes, cap=class_weight_cap)
        criterion = nn.CrossEntropyLoss(weight=weights.to(dev), label_smoothing=label_smoothing)
    elif mitigation == "group_weighted":
        group_weights = compute_tone_group_weights(train_ds)
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    elif mitigation == "balanced_sampler":
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        # Compute sample weights
        if hasattr(train_ds, "lesions") and "diagnosis" in train_ds.lesions.columns:
            labels = [MILK10K_CLASS_TO_IDX[d] for d in train_ds.lesions["diagnosis"]]
        elif hasattr(train_ds, "indices") and hasattr(train_ds, "dataset") and hasattr(train_ds.dataset, "lesions"):
            labels = [MILK10K_CLASS_TO_IDX[train_ds.dataset.lesions.iloc[i]["diagnosis"]] for i in train_ds.indices]
        else:
            labels = [b["label"].item() if isinstance(b["label"], torch.Tensor) else int(b["label"]) for b in train_ds]
        counts = np.bincount(labels, minlength=num_classes)
        sample_weights = [1.0 / max(counts[l], 1) for l in labels]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        dataloaders["train"] = DataLoader(
            train_ds,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=0,
            collate_fn=dataloaders["train"].collate_fn,
        )
    else:  # none
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    val_criterion = nn.CrossEntropyLoss()

    # Build Model
    model = build_model(
        model_type=model_name,
        num_classes=num_classes,
        backbone="efficientnet_b0",
        pretrained=True,
        dropout=0.2,
        fusion_type=fusion_type,
        tabular_dim=info["tabular_dim"],
    ).to(dev)

    # Parameter groups for differential learning rates
    param_groups = get_parameter_groups(
        model=model,
        lr_encoder=lr_encoder,
        lr_head=lr_head,
        weight_decay=weight_decay,
    )
    optimizer = torch.optim.AdamW(param_groups)
    scheduler = build_warmup_cosine_scheduler(optimizer, warmup_epochs=warmup_epochs, total_epochs=epochs)

    best_val_f1 = -1.0
    epochs_without_improvement = 0
    history: List[Dict[str, Any]] = []

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        train_loss, train_acc, train_f1 = train_one_epoch_milk10k(
            model=model,
            dataloader=dataloaders["train"],
            model_name=model_name,
            criterion=criterion,
            optimizer=optimizer,
            device=dev,
            mitigation=mitigation,
            group_weights=group_weights,
        )

        val_loss, val_acc, val_f1, _, _, _ = evaluate_milk10k_loader(
            model=model,
            dataloader=dataloaders["val"],
            model_name=model_name,
            criterion=val_criterion,
            device=dev,
            metric_indices=eligible_indices,
        )

        scheduler.step()
        epoch_time = time.time() - t0

        if dev.type == "mps":
            torch.mps.empty_cache()

        is_best = val_f1 > best_val_f1
        if is_best:
            best_val_f1 = val_f1
            epochs_without_improvement = 0
            torch.save(model.state_dict(), best_ckpt_path)
        else:
            epochs_without_improvement += 1

        history.append({
            "epoch": epoch,
            "epoch_time_sec": epoch_time,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "train_f1": train_f1,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_f1": val_f1,
        })

        best_mark = " [*BEST]" if is_best else ""
        epoch_log_str = (
            f"Fold {fold} | Epoch {epoch:02d}/{epochs:02d} | "
            f"Time: {epoch_time:.2f}s | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc*100:.1f}% F1: {train_f1:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc*100:.1f}% Val-F1: {val_f1:.4f}{best_mark}"
        )
        print(epoch_log_str, flush=True)
        if logger_instance:
            logger_instance.info(epoch_log_str)

        if epochs_without_improvement >= patience:
            stop_msg = f"Fold {fold} | Early stopping triggered at epoch {epoch:02d} (patience: {patience}, best val F1: {best_val_f1:.4f})"
            print(f"[*] {stop_msg}", flush=True)
            if logger_instance:
                logger_instance.info(stop_msg)
            break

    # Load best checkpoint for held-out test evaluation
    if best_ckpt_path.exists():
        model.load_state_dict(torch.load(best_ckpt_path, map_location=dev))
    else:
        torch.save(model.state_dict(), best_ckpt_path)

    # Evaluate on held-out test fold
    _, _, test_f1, preds_arr, targets_arr, probs_arr = evaluate_milk10k_loader(
        model=model,
        dataloader=dataloaders["test"],
        model_name=model_name,
        criterion=val_criterion,
        device=dev,
    )

    test_ds = dataloaders["test"].dataset
    lesions_test = test_ds.lesions

    pred_records: List[Dict[str, Any]] = []
    for i in range(len(test_ds)):
        row = lesions_test.iloc[i]
        rec = {
            "lesion_id": str(row["lesion_id"]),
            "fold": fold,
            "y_true": MILK10K_IDX_TO_CLASS[targets_arr[i]],
            "y_pred": MILK10K_IDX_TO_CLASS[preds_arr[i]],
            "tone_group": str(row["tone_group"]),
            "skin_tone_class": int(row["skin_tone_class"]),
        }
        for c_idx, c_name in enumerate(MILK10K_CLASSES):
            rec[f"prob_{c_name}"] = float(probs_arr[i, c_idx])
        pred_records.append(rec)

    pred_df = pd.DataFrame(pred_records)
    return pred_df, best_val_f1, history


def run_milk10k_training(
    model: str = "two_image_tabular_fusion",
    fold: Union[int, str] = 0,
    epochs: int = 12,
    img_size: int = 128,
    run_name: Optional[str] = None,
    mitigation: str = "class_weighted",
    batch_size: int = 32,
    lr_encoder: float = 1e-4,
    lr_head: float = 1e-3,
    lr: float = 3e-4,
    weight_decay: float = 0.01,
    label_smoothing: float = 0.05,
    warmup_epochs: int = 1,
    patience: int = 4,
    min_val_samples_per_class: int = 5,
    class_weight_cap: float = 10.0,
    fusion_type: str = "concat",
    augmentation: str = "standard",
    use_monet: bool = False,
    seed: int = 42,
    reports_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Main entry point for MILK10k training.
    Supports single-fold evaluation (e.g. --fold 0) or 5-fold CV (e.g. --fold all).
    Resumable loop: skips already completed folds if predictions file exists.
    """
    start_total_time = time.time()
    set_seed(seed)
    dev = get_device()

    actual_run_name = run_name or f"{model}_{mitigation}_{fold}_{img_size}px"
    rep_dir = reports_dir or (VISHWAS_ROOT / "reports" / actual_run_name)
    rep_dir.mkdir(parents=True, exist_ok=True)

    # Configure file logging to logs/<run-name>.log
    logs_dir = VISHWAS_ROOT / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_file = logs_dir / f"{actual_run_name}.log"

    run_logger = logging.getLogger(f"vishwas.run.{actual_run_name}")
    run_logger.setLevel(logging.INFO)
    if not any(isinstance(h, logging.FileHandler) and Path(h.baseFilename) == log_file.resolve() for h in run_logger.handlers):
        fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        fh.setFormatter(logging.Formatter("[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
        run_logger.addHandler(fh)

    init_msg = (
        f"Starting run '{actual_run_name}' | Model: {model} | Fold: {fold} | Epochs: {epochs} | "
        f"Image Size: {img_size} | Mitigation: {mitigation} | LR Encoder: {lr_encoder} | LR Head: {lr_head}"
    )
    print(init_msg, flush=True)
    run_logger.info(init_msg)

    folds_to_run: List[int] = list(range(5)) if str(fold).lower() == "all" else [int(fold)]
    all_fold_preds: List[pd.DataFrame] = []
    total_epoch_times: List[float] = []

    for f in folds_to_run:
        fold_pred_path = rep_dir / f"fold_{f}_predictions.csv"
        best_ckpt_path = VISHWAS_ROOT / "checkpoints" / f"best_{actual_run_name}_fold{f}.pt"

        # Check if fold already finished (resumable loop)
        if fold_pred_path.exists() and best_ckpt_path.exists():
            try:
                f_preds = pd.read_csv(fold_pred_path)
                if len(f_preds) >= 1000:
                    resume_msg = f"[RESUME] Fold {f} already completed ({len(f_preds)} predictions). Skipping training."
                    print(f"\n{'='*70}\n {resume_msg}\n{'='*70}", flush=True)
                    run_logger.info(resume_msg)
                    all_fold_preds.append(f_preds)
                    continue
            except Exception as e:
                logger.warning(f"Error reading existing {fold_pred_path}: {e}. Retraining fold {f}...")

        f_preds, _, history = train_milk10k_fold(
            fold=f,
            model_name=model,
            epochs=epochs,
            img_size=img_size,
            run_name=actual_run_name,
            mitigation=mitigation,
            batch_size=batch_size,
            lr_encoder=lr_encoder,
            lr_head=lr_head,
            lr=lr,
            weight_decay=weight_decay,
            label_smoothing=label_smoothing,
            warmup_epochs=warmup_epochs,
            patience=patience,
            min_val_samples_per_class=min_val_samples_per_class,
            class_weight_cap=class_weight_cap,
            fusion_type=fusion_type,
            augmentation=augmentation,
            use_monet=use_monet,
            seed=seed,
            device=dev,
            logger_instance=run_logger,
        )
        # Save individual fold predictions immediately
        f_preds.to_csv(fold_pred_path, index=False)
        all_fold_preds.append(f_preds)
        total_epoch_times.extend([h["epoch_time_sec"] for h in history])

    # Pool predictions
    pooled_df = pd.concat(all_fold_preds, axis=0).reset_index(drop=True)
    pred_path = rep_dir / "predictions.csv"
    pooled_df.to_csv(pred_path, index=False)
    save_msg = f"Saved {len(pooled_df)} pooled predictions to: {pred_path}"
    print(f"\n{save_msg}", flush=True)
    run_logger.info(save_msg)

    # Comprehensive Fairness & Subgroup Evaluation
    eval_results = evaluate_milk10k_predictions(
        predictions_df=pooled_df,
        output_dir=rep_dir,
        classes=MILK10K_CLASSES,
        n_bootstrap=1000,
        seed=seed,
    )

    total_runtime = time.time() - start_total_time
    avg_epoch_time = float(np.mean(total_epoch_times)) if total_epoch_times else 0.0

    # Log to reports/experiments.csv
    exp_log = VISHWAS_ROOT / "reports" / "experiments.csv"
    exp_log.parent.mkdir(parents=True, exist_ok=True)

    ov = eval_results["overall"]
    disp = eval_results["disparity"]

    exp_entry = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "run_name": actual_run_name,
        "dataset": "milk10k",
        "model": model,
        "fold": str(fold),
        "epochs": epochs,
        "img_size": img_size,
        "mitigation": mitigation,
        "fusion_type": fusion_type,
        "augmentation": augmentation,
        "use_monet": use_monet,
        "overall_acc": round(ov["accuracy"], 4),
        "headline_macro_f1": round(ov["headline_macro_f1"], 4),
        "full_macro_f1": round(ov["full_macro_f1"], 4),
        "worst_group_f1": round(disp["worst_f1"], 4),
        "f1_gap": round(disp["f1_gap"], 4),
        "avg_epoch_time_sec": round(avg_epoch_time, 2),
        "total_runtime_sec": round(total_runtime, 2),
        "seed": seed,
    }

    exp_df = pd.DataFrame([exp_entry])
    if exp_log.exists():
        exp_df.to_csv(exp_log, mode="a", header=False, index=False)
    else:
        exp_df.to_csv(exp_log, index=False)

    summary_str = (
        f"\n{'='*70}\n"
        f"                 MILK10K EVALUATION SUMMARY\n"
        f"{'='*70}\n"
        f"Run Name           : {actual_run_name}\n"
        f"Model Architecture : {model}\n"
        f"Evaluated Folds    : {fold}\n"
        f"Total Lesions      : {len(pooled_df):,}\n"
        f"Average Epoch Time : {avg_epoch_time:.2f} s\n"
        f"Total Runtime      : {total_runtime:.2f} s\n"
        f"{'-'*70}\n"
        f"Overall Accuracy   : {ov['accuracy']*100:.2f}% (95% CI: [{ov['cis']['accuracy'][0]*100:.1f}%, {ov['cis']['accuracy'][1]*100:.1f}%])\n"
        f"Headline Macro-F1  : {ov['headline_macro_f1']:.4f} (95% CI: [{ov['cis']['macro_f1'][0]:.4f}, {ov['cis']['macro_f1'][1]:.4f}])\n"
        f"Balanced Accuracy  : {ov['balanced_accuracy']*100:.2f}% (95% CI: [{ov['cis']['balanced_accuracy'][0]*100:.1f}%, {ov['cis']['balanced_accuracy'][1]*100:.1f}%])\n"
        f"Macro-AUC (OvR)    : {ov['macro_auc']:.4f}\n"
        f"Calibration (ECE)  : {ov['ece']:.4f} (95% CI: [{ov['cis'].get('ece', (0.0, 1.0))[0]:.4f}, {ov['cis'].get('ece', (0.0, 1.0))[1]:.4f}])\n"
        f"{'-'*70}\n"
        f"Per Tone Group Performance (G1={{0,1,2}}, G2={{3}}, G3={{4,5}}):\n"
    )
    for g, gm in eval_results["per_group"].items():
        ci = gm["cis"]["macro_f1"]
        ci_e = gm["cis"].get("ece", (0.0, 1.0))
        summary_str += (
            f"  - Group {g:<7}: n={gm['n_lesions']:<4} | Macro-F1: {gm['headline_macro_f1']:.4f} [{ci[0]:.4f}, {ci[1]:.4f}] | "
            f"Acc: {gm['accuracy']*100:.1f}% | ECE: {gm['ece']:.4f} [{ci_e[0]:.4f}, {ci_e[1]:.4f}]\n"
        )
    summary_str += (
        f"Worst Group        : {disp['worst_group']} (F1: {disp['worst_f1']:.4f})\n"
        f"Disparity Gap (F1) : {disp['f1_gap']:.4f}\n"
        f"{'='*70}\n"
    )
    print(summary_str, flush=True)
    run_logger.info(summary_str)

    return eval_results
