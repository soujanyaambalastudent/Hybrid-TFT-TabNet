# ============================================================
#  train.py
#  Trains ONE model (the proposed hybrid, or any of the 7
#  baselines) on ONE fold of splits.json, per its config file.
#  Also supports the two frozen/ablation variants needed for
#  Table 6 (A5: frozen TabNet) via --freeze_tabnet.
#
#  Usage (Colab or local):
#    python train.py --model proposed --fold 1 \
#        --config configs/proposed.json \
#        --splits /content/ptbxl_meta/splits.json \
#        --labels /content/ptbxl_meta/ischemic_labels.csv
#
#    python train.py --model cnn_1d --fold 1 \
#        --config configs/baselines/cnn_1d.json ...
#
#  Requires the ECG waveforms and clinical features to already be
#  preprocessed via preprocessing.py and cached to disk (this
#  script expects a get_ecg_tensor(record_id) / get_clinical_vector
#  (record_id) pair of lookup functions -- wire these to wherever
#  you cache the preprocessed arrays, e.g. a memory-mapped .npy
#  file or per-record .npz files, since PTB-XL's raw waveforms
#  (~2 GB) are too large to bundle in this repository).
# ============================================================

import argparse
import json
import time

import torch
import torch.utils.data as data

from seed_utils import fix_all_seeds
from models.fusion_model import HybridTFTTabNet, label_smoothed_bce_loss
from models.baselines import BASELINE_REGISTRY
from models.ablation_fusion import EarlyFusionModel, LateFusionModel

# Model names available only as ablation configurations (Table 6, A3/A4),
# not as baselines compared in Tables 4/8/9.
ABLATION_ONLY_MODELS = ("early_fusion", "late_fusion")


class ECGClinicalDataset(data.Dataset):
    """Wires record IDs to preprocessed tensors. Replace
    `get_ecg_tensor` / `get_clinical_vector` with your actual
    on-disk cache lookups -- see preprocessing.py."""

    def __init__(self, record_ids, labels_by_id, get_ecg_tensor, get_clinical_vector):
        self.record_ids = record_ids
        self.labels_by_id = labels_by_id
        self.get_ecg_tensor = get_ecg_tensor
        self.get_clinical_vector = get_clinical_vector

    def __len__(self):
        return len(self.record_ids)

    def __getitem__(self, idx):
        rid = self.record_ids[idx]
        ecg = torch.as_tensor(self.get_ecg_tensor(rid), dtype=torch.float32)
        clinical = torch.as_tensor(self.get_clinical_vector(rid), dtype=torch.float32)
        label = torch.tensor(float(self.labels_by_id[rid]), dtype=torch.float32)
        return ecg, clinical, label


def build_model(model_name, config, freeze_tabnet=False):
    if model_name == "proposed":
        return HybridTFTTabNet(
            ecg_config=config["architecture"]["ecg_branch"],
            clinical_config=config["architecture"]["clinical_branch"],
            fusion_hidden_dim=config["architecture"]["fusion_hidden_dim"],
            freeze_tabnet=freeze_tabnet,
        )
    if model_name == "early_fusion":
        # Ablation A3 (Table 6): single shared encoder, clinical vector
        # concatenated to the raw ECG sample before any encoding.
        return EarlyFusionModel(**config["architecture"])
    if model_name == "late_fusion":
        # Ablation A4 (Table 6): independent TFT/TabNet heads, combined
        # only via probability-averaging at prediction time.
        return LateFusionModel(
            ecg_config=config["architecture"]["ecg_branch"],
            clinical_config=config["architecture"]["clinical_branch"],
        )
    if model_name not in BASELINE_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}")
    return BASELINE_REGISTRY[model_name](**config["architecture"])


def cosine_warmup_lr(step, total_steps, warmup_steps, base_lr):
    if step < warmup_steps:
        return base_lr * step / max(1, warmup_steps)
    import math
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return base_lr * 0.5 * (1 + math.cos(math.pi * progress))


def train_one_fold(model_name, config, fold_num, splits, labels_by_id,
                    get_ecg_tensor, get_clinical_vector, device="cuda",
                    freeze_tabnet=False, checkpoint_dir="./checkpoints"):
    fix_all_seeds(config.get("seed", 42))

    fold_key = f"fold{fold_num}"
    train_ids = splits["folds"][fold_key]["train"]
    val_ids = splits["folds"][fold_key]["val"]

    train_ds = ECGClinicalDataset(train_ids, labels_by_id, get_ecg_tensor, get_clinical_vector)
    val_ds = ECGClinicalDataset(val_ids, labels_by_id, get_ecg_tensor, get_clinical_vector)

    tcfg = config["training"]
    train_loader = data.DataLoader(train_ds, batch_size=tcfg["batch_size"], shuffle=True)
    val_loader = data.DataLoader(val_ds, batch_size=tcfg["batch_size"], shuffle=False)

    model = build_model(model_name, config, freeze_tabnet=freeze_tabnet).to(device)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=tcfg["lr"], weight_decay=tcfg["weight_decay"],
    )

    max_epochs = tcfg["max_epochs"]
    warmup_epochs = tcfg.get("warmup_epochs", 5)
    patience = tcfg.get("early_stopping_patience", 10)
    eps = tcfg.get("label_smoothing_eps", 0.0)

    steps_per_epoch = max(1, len(train_loader))
    total_steps = steps_per_epoch * max_epochs
    warmup_steps = steps_per_epoch * warmup_epochs

    best_val_auc = -1.0
    best_val_metrics = None
    epochs_without_improvement = 0
    global_step = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        epoch_start = time.time()
        for ecg, clinical, labels in train_loader:
            ecg, clinical, labels = ecg.to(device), clinical.to(device), labels.to(device)

            lr = cosine_warmup_lr(global_step, total_steps, warmup_steps, tcfg["lr"])
            for g in optimizer.param_groups:
                g["lr"] = lr

            optimizer.zero_grad()
            logits, mask_loss = model(ecg, clinical)
            loss = label_smoothed_bce_loss(logits, labels, eps=eps, mask_loss=mask_loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg["grad_clip_norm"])
            optimizer.step()
            global_step += 1

        val_metrics = evaluate_fold_metrics(model, val_loader, device)
        val_auc = val_metrics["auc"]
        print(f"[{model_name} fold {fold_num}] epoch {epoch}/{max_epochs}  "
              f"val_auc={val_auc:.4f}  ({time.time() - epoch_start:.1f}s)")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_val_metrics = val_metrics
            epochs_without_improvement = 0
            import os
            os.makedirs(checkpoint_dir, exist_ok=True)
            torch.save(model.state_dict(),
                       f"{checkpoint_dir}/{model_name}_fold{fold_num}_best.pt")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch} (patience={patience})")
                break

    return best_val_auc, best_val_metrics


def evaluate_fold_metrics(model, loader, device):
    """Computes the full Table 4 metric set (accuracy, F1, AUC-ROC,
    sensitivity, specificity) on one fold's validation set -- not just
    AUC -- so a single training run produces everything Tables 4, 6,
    and 8 need, logged via --metrics_log (see main() below)."""
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix

    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for ecg, clinical, labels in loader:
            ecg, clinical = ecg.to(device), clinical.to(device)
            logits, _ = model(ecg, clinical)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs.tolist())
            all_labels.extend(labels.numpy().tolist())

    y_true = all_labels
    y_prob = all_probs
    y_pred = [1 if p >= 0.5 else 0 for p in y_prob]

    auc = roc_auc_score(y_true, y_prob)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "auc": auc,
        "sensitivity": tp / (tp + fn) if (tp + fn) > 0 else float("nan"),
        "specificity": tn / (tn + fp) if (tn + fp) > 0 else float("nan"),
    }


def log_fold_metrics(metrics_log_path, model_name, fold_num, metrics, freeze_tabnet=False):
    """Appends one row to the shared CSV log so evaluate.py's ensembling,
    the Table 8 sweep, and significance_testing.py's Wilcoxon comparison
    all have a single, real source of per-fold metrics to read from,
    instead of only stdout."""
    import csv
    import os

    header = ["model", "fold", "freeze_tabnet", "accuracy", "f1", "auc",
              "sensitivity", "specificity"]
    row = [model_name, fold_num, freeze_tabnet, metrics["accuracy"], metrics["f1"],
           metrics["auc"], metrics["sensitivity"], metrics["specificity"]]

    write_header = not os.path.exists(metrics_log_path)
    os.makedirs(os.path.dirname(metrics_log_path) or ".", exist_ok=True)
    with open(metrics_log_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        writer.writerow(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True,
                         choices=["proposed"] + list(ABLATION_ONLY_MODELS) + list(BASELINE_REGISTRY.keys()))
    parser.add_argument("--fold", type=int, required=True, choices=[1, 2, 3, 4, 5])
    parser.add_argument("--config", required=True)
    parser.add_argument("--splits", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--freeze_tabnet", action="store_true",
                         help="Ablation A5 (Table 6): freeze the TabNet branch "
                              "(only meaningful with --model proposed).")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--checkpoint_dir", default="./checkpoints")
    parser.add_argument("--metrics_log", default="./artifacts/cv_metrics.csv",
                         help="CSV that this fold's best validation metrics are appended "
                              "to -- the single source Tables 4/6/8 and "
                              "significance_testing.py's Wilcoxon comparison read from.")
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)
    with open(args.splits) as f:
        splits = json.load(f)

    import pandas as pd
    labels_df = pd.read_csv(args.labels)
    labels_by_id = dict(zip(labels_df["record_id"], labels_df["ischemic_label"]))

    from data_cache import get_ecg_tensor, get_clinical_vector

    best_val_auc, best_val_metrics = train_one_fold(
        args.model, config, args.fold, splits, labels_by_id,
        get_ecg_tensor, get_clinical_vector,
        device=args.device, freeze_tabnet=args.freeze_tabnet,
        checkpoint_dir=args.checkpoint_dir,
    )
    print(f"\nBest validation AUC for {args.model} fold {args.fold}: {best_val_auc:.4f}")

    log_fold_metrics(args.metrics_log, args.model, args.fold, best_val_metrics,
                      freeze_tabnet=args.freeze_tabnet)
    print(f"Logged fold metrics -> {args.metrics_log}")


if __name__ == "__main__":
    main()
