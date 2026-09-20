# ============================================================
#  computational_profile.py
#  Reproduces Table 9 (Section 6.8): trainable parameter counts,
#  per-epoch training time, and per-sample inference latency for the
#  proposed model and all seven baselines.
#
#  Until now, these numbers lived only as hand-typed
#  "reference_params_million" fields in the baseline config JSONs --
#  no script anywhere in the repo actually instantiated the models and
#  measured them. This script does, using synthetic input tensors of
#  the correct shape, so it needs no cached dataset and no GPU to run
#  (though timings will differ from an A100 GPU, obviously -- see the
#  printed device name).
#
#  Usage:
#    python computational_profile.py --device cpu --batch_size 256
# ============================================================

import argparse
import json
import time

import torch

from models.fusion_model import HybridTFTTabNet, label_smoothed_bce_loss
from models.baselines import BASELINE_REGISTRY
from models.ablation_fusion import EarlyFusionModel, LateFusionModel

N_LEADS = 12
SEQ_LEN = 5000
CLINICAL_DIM = 25


def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def time_forward_backward(model, ecg, clinical, labels, device, n_repeats=3):
    """Average wall-clock time for one forward + backward pass, used as
    a per-batch proxy for Table 9's "Train Time/Epoch" column (multiply
    by n_batches_per_epoch for a full-epoch estimate)."""
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    times = []
    for _ in range(n_repeats):
        if device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        optimizer.zero_grad()
        logits, mask_loss = model(ecg, clinical)
        loss = label_smoothed_bce_loss(logits, labels, mask_loss=mask_loss)
        loss.backward()
        optimizer.step()
        if device == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    return min(times)  # min, not mean, to reduce first-call/warmup noise


def time_inference_latency(model, ecg_single, clinical_single, device, n_repeats=20):
    """Per-sample inference latency (batch size 1), matching Table 9's
    "Infer. Latency (ms)" column."""
    model.eval()
    times = []
    with torch.no_grad():
        for _ in range(n_repeats):
            if device == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            model(ecg_single, clinical_single)
            if device == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - start)
    return min(times) * 1000  # ms


def build_all_models():
    models = {
        "proposed": HybridTFTTabNet(
            ecg_config=dict(n_leads=N_LEADS, seq_len=SEQ_LEN, embed_dim=64,
                             lstm_layers=2, attn_heads=8, dropout=0.1),
            clinical_config=dict(input_dim=CLINICAL_DIM, embed_dim=64, n_steps=5, n_shared=2),
            fusion_hidden_dim=64,
        ),
        "early_fusion": EarlyFusionModel(
            n_leads=N_LEADS, seq_len=SEQ_LEN, clinical_dim=CLINICAL_DIM, embed_dim=64,
            lstm_layers=2, attn_heads=8, dropout=0.3,
        ),
        "late_fusion": LateFusionModel(
            ecg_config=dict(n_leads=N_LEADS, seq_len=SEQ_LEN, embed_dim=64,
                             lstm_layers=2, attn_heads=8, dropout=0.1),
            clinical_config=dict(input_dim=CLINICAL_DIM, embed_dim=64, n_steps=5, n_shared=2),
        ),
    }
    for name, cls in BASELINE_REGISTRY.items():
        models[name] = cls()
    return models


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--out", default="./artifacts/table9_computational_profile.json")
    args = parser.parse_args()
    device = args.device

    print(f"Device: {device}  (Table 9 in the manuscript was measured on a single "
          f"NVIDIA A100 GPU -- absolute timings here will differ if run on CPU "
          f"or a different GPU; relative ordering across models is still meaningful.)")

    B = args.batch_size
    ecg = torch.randn(B, N_LEADS, SEQ_LEN, device=device)
    clinical = torch.randn(B, CLINICAL_DIM, device=device)
    labels = torch.randint(0, 2, (B,), device=device).float()
    ecg1 = torch.randn(1, N_LEADS, SEQ_LEN, device=device)
    clinical1 = torch.randn(1, CLINICAL_DIM, device=device)

    results = {}
    for name, model in build_all_models().items():
        model = model.to(device)
        n_params = count_trainable_params(model)
        batch_time = time_forward_backward(model, ecg, clinical, labels, device)
        latency_ms = time_inference_latency(model, ecg1, clinical1, device)

        results[name] = {
            "params_million": round(n_params / 1e6, 3),
            "measured_batch_fwd_bwd_time_s": round(batch_time, 4),
            "measured_inference_latency_ms": round(latency_ms, 4),
        }
        print(f"  {name:15s}  params={n_params/1e6:6.2f}M  "
              f"batch_fwd_bwd={batch_time*1000:7.1f}ms  "
              f"infer_latency={latency_ms:6.2f}ms")

    import os
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"device": device, "batch_size": B, "results": results}, f, indent=2)
    print(f"\nSaved -> {args.out}")
    print("\nNOTE: 'Train Time/Epoch' in Table 9 additionally requires the number of "
          "batches per epoch (train-pool size / batch_size); multiply "
          "measured_batch_fwd_bwd_time_s by that count for a full-epoch estimate.")


if __name__ == "__main__":
    main()
