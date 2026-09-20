# ============================================================
#  models/fusion_model.py
#  The proposed hybrid model: concatenates the 64-dim TFT
#  embedding and 64-dim TabNet embedding into a 128-dim joint
#  representation, passes it through a 2-layer MLP (128 -> 64,
#  ReLU, batch norm, dropout 0.3), then a final linear + sigmoid
#  layer. Label smoothing (eps=0.05) is applied to the BCE loss,
#  per Section IV.D.
#
#  Also supports "frozen TabNet" mode, used for ablation A5
#  (Table 6): TabNet branch is loaded but its parameters are
#  frozen, so only the TFT branch and fusion head train.
# ============================================================

import torch
import torch.nn as nn

from models.tft_branch import TFTBranch
from models.tabnet_branch import TabNetBranch


class FusionHead(nn.Module):
    def __init__(self, in_dim=128, hidden_dim=64, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, joint_embedding):
        logit = self.net(joint_embedding).squeeze(-1)  # (B,)
        return logit


class HybridTFTTabNet(nn.Module):
    def __init__(self, ecg_config=None, clinical_config=None,
                 fusion_hidden_dim=64, freeze_tabnet=False):
        super().__init__()
        ecg_config = ecg_config or {}
        clinical_config = clinical_config or {}

        self.tft = TFTBranch(**ecg_config)
        self.tabnet = TabNetBranch(**clinical_config)
        self.fusion = FusionHead(in_dim=self.tft.embed_dim + self.tabnet.embed_dim,
                                  hidden_dim=fusion_hidden_dim)

        if freeze_tabnet:
            for p in self.tabnet.parameters():
                p.requires_grad = False

    def forward(self, ecg, clinical_features):
        tft_emb = self.tft(ecg)                                    # (B, 64)
        tabnet_emb, mask_loss = self.tabnet(clinical_features)      # (B, 64)
        joint = torch.cat([tft_emb, tabnet_emb], dim=-1)             # (B, 128)
        logit = self.fusion(joint)
        return logit, mask_loss


def label_smoothed_bce_loss(logits, targets, eps=0.05, mask_loss=None,
                             mask_loss_weight=1e-3):
    """Binary cross-entropy with label smoothing, plus TabNet's
    sparsity mask-loss regularization term (standard TabNet training
    practice), matching Section IV.D."""
    targets_smoothed = targets * (1 - eps) + 0.5 * eps
    bce = nn.functional.binary_cross_entropy_with_logits(logits, targets_smoothed)
    if mask_loss is not None:
        bce = bce + mask_loss_weight * mask_loss
    return bce
