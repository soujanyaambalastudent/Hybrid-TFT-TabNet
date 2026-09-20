# ============================================================
#  models/ablation_fusion.py
#  Table 6 lists six ablation configurations (A1-A6). A1 (TFT-only),
#  A2 (TabNet-only), A5 (intermediate fusion, frozen TabNet) and A6
#  (the full proposed model) all had working code already -- A1/A2
#  reuse models/baselines.py, A5/A6 reuse models/fusion_model.py's
#  --freeze_tabnet flag. A3 (early fusion) and A4 (late fusion) had
#  NO model implementation anywhere in the repository; this file adds
#  them so Table 6 is fully reproducible end-to-end.
#
#  Both reuse the same TFTBranch / TabNetBranch building blocks as the
#  proposed model, so the comparison isolates the fusion STRATEGY
#  rather than confounding it with unrelated architecture changes.
# ============================================================

import torch
import torch.nn as nn

from models.tft_branch import PositionalEncoding, GatedResidualNetwork, VariableSelectionNetwork
from models.tabnet_branch import TabNetBranch


class EarlyFusionModel(nn.Module):
    """Ablation A3: "raw concatenation of ECG features and clinical
    features before encoding" (Table 6). The 25-dim clinical vector is
    broadcast across every time step and concatenated to each lead's
    raw sample value at the FIRST linear projection, i.e. before any
    temporal encoding happens -- there is no separate TabNet encoder
    in this configuration; a single shared encoder (structurally
    identical to the TFT branch, but with a wider input) processes the
    fused per-timestep representation directly into one classification
    head. This is what makes it "early"/feature-level fusion, as
    opposed to the proposed model's intermediate fusion of two
    independently-encoded 64-dim embeddings.
    """

    def __init__(self, n_leads=12, seq_len=5000, clinical_dim=25, embed_dim=64,
                 lstm_layers=2, attn_heads=8, dropout=0.3, attn_pool_size=25):
        super().__init__()
        self.n_leads = n_leads
        self.embed_dim = embed_dim
        d_model = embed_dim * 2

        if seq_len % attn_pool_size != 0:
            raise ValueError(
                f"seq_len ({seq_len}) must be divisible by attn_pool_size ({attn_pool_size})."
            )

        self.input_proj = nn.Linear(1 + clinical_dim, embed_dim)
        self.pos_enc = PositionalEncoding(embed_dim, max_len=seq_len)
        self.vsn = VariableSelectionNetwork(embed_dim)
        self.lstm = nn.LSTM(
            input_size=embed_dim, hidden_size=embed_dim, num_layers=lstm_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        # Same tractability fix as TFTBranch (see models/tft_branch.py's
        # docstring): full self-attention over T=5000 raw time steps is
        # infeasible at batch_size=256 (O(T^2) attention-weight memory),
        # so attention runs on the LSTM output downsampled by this factor.
        self.attn_pool = nn.AvgPool1d(kernel_size=attn_pool_size, stride=attn_pool_size)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=attn_heads, batch_first=True, dropout=dropout
        )
        self.grn = GatedResidualNetwork(d_model, dropout=dropout)
        self.out_proj = nn.Linear(d_model, embed_dim)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(embed_dim, 1))

    def _encode_one_lead(self, lead_signal, clinical_features):
        # lead_signal: (B, T); clinical_features: (B, clinical_dim)
        B, T = lead_signal.shape
        clinical_bcast = clinical_features.unsqueeze(1).expand(B, T, -1)
        x = torch.cat([lead_signal.unsqueeze(-1), clinical_bcast], dim=-1)  # (B, T, 1+clinical_dim)
        x = self.input_proj(x)
        x = self.pos_enc(x)
        x = self.vsn(x)
        x, _ = self.lstm(x)
        x = self.attn_pool(x.transpose(1, 2)).transpose(1, 2)  # (B, T', 2*embed_dim)
        attn_out, _ = self.attn(x, x, x)
        x = self.grn(attn_out)
        x = x.mean(dim=1)
        return self.out_proj(x)

    def forward(self, ecg, clinical_features):
        lead_embeddings = [
            self._encode_one_lead(ecg[:, lead, :], clinical_features)
            for lead in range(self.n_leads)
        ]
        pooled = torch.stack(lead_embeddings, dim=1).mean(dim=1)
        logit = self.head(pooled).squeeze(-1)
        return logit, None


class LateFusionModel(nn.Module):
    """Ablation A4: "late fusion (independent predictions averaged)"
    (Table 6). The TFT and TabNet branches are each given their OWN
    classification head -- there is no shared fusion layer, so the two
    branches never see each other's representation during training.
    At inference (and for the loss), the two branches' predicted
    probabilities are averaged, matching the manuscript's description
    exactly, rather than averaging logits (which would not correspond
    to "predictions averaged").

    To keep this drop-in compatible with the rest of the training/eval
    code (which expects `forward()` to return a single logit tensor),
    the averaged probability is converted back to a logit via the
    inverse sigmoid; this is differentiable, so both branch heads still
    receive independent gradients during backprop -- it is exactly
    equivalent to training on the averaged probability directly.
    """

    def __init__(self, ecg_config=None, clinical_config=None, dropout=0.3):
        super().__init__()
        from models.tft_branch import TFTBranch

        ecg_config = ecg_config or {}
        clinical_config = clinical_config or {}

        self.tft = TFTBranch(**ecg_config)
        self.tabnet = TabNetBranch(**clinical_config)
        self.tft_head = nn.Sequential(nn.Dropout(dropout), nn.Linear(self.tft.embed_dim, 1))
        self.tabnet_head = nn.Sequential(nn.Dropout(dropout), nn.Linear(self.tabnet.embed_dim, 1))

    def forward(self, ecg, clinical_features):
        tft_emb = self.tft(ecg)
        tabnet_emb, mask_loss = self.tabnet(clinical_features)

        tft_logit = self.tft_head(tft_emb).squeeze(-1)
        tabnet_logit = self.tabnet_head(tabnet_emb).squeeze(-1)

        avg_prob = 0.5 * (torch.sigmoid(tft_logit) + torch.sigmoid(tabnet_logit))
        avg_prob = avg_prob.clamp(1e-6, 1 - 1e-6)
        combined_logit = torch.log(avg_prob / (1 - avg_prob))
        return combined_logit, mask_loss
