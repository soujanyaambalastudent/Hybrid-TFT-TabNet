# ============================================================
#  models/baselines.py
#  The 7 baseline architectures compared in Tables 4/8/9:
#    CNN (1-D), ResNet-34 (1-D), LSTM, Bi-LSTM+Attention,
#    Transformer, TabNet-only, TFT-only.
#  TabNet-only and TFT-only reuse the branch modules directly
#  with a lightweight classification head (they are literally
#  ablations A2 / A1 of the proposed model -- see Table 6 -- so
#  reusing the same branch code guarantees consistency between
#  the baseline table and the ablation table).
#
#  Hyperparameters for each are read from configs/baselines/*.json
#  at training time (see train.py) -- this file defines only the
#  architectures themselves.
# ============================================================

import torch
import torch.nn as nn

from models.tft_branch import TFTBranch
from models.tabnet_branch import TabNetBranch


# ---------------------------------------------------------------
# 1. CNN (1-D)
# ---------------------------------------------------------------
class CNN1D(nn.Module):
    def __init__(self, n_leads=12, n_filters=(32, 64, 128), kernel_size=7, dropout=0.3):
        super().__init__()
        layers = []
        in_ch = n_leads
        for out_ch in n_filters:
            layers += [
                nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size // 2),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(),
                nn.MaxPool1d(2),
            ]
            in_ch = out_ch
        self.conv = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_ch, 1))

    def forward(self, ecg, clinical_features=None):
        x = self.conv(ecg)
        x = self.pool(x).squeeze(-1)
        return self.head(x).squeeze(-1), None


# ---------------------------------------------------------------
# 2. ResNet-34 (1-D)
# ---------------------------------------------------------------
class ResBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU()
        self.downsample = None
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch),
            )

    def forward(self, x):
        identity = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class ResNet34_1D(nn.Module):
    """34-layer residual network adapted to 1-D ECG signals, following
    the standard ResNet-34 block layout [3,4,6,3]."""

    def __init__(self, n_leads=12, base_ch=64, dropout=0.3):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(n_leads, base_ch, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(base_ch),
            nn.ReLU(),
            nn.MaxPool1d(3, stride=2, padding=1),
        )
        block_counts = [3, 4, 6, 3]
        channels = [base_ch, base_ch * 2, base_ch * 4, base_ch * 8]
        layers = []
        in_ch = base_ch
        for stage, (n_blocks, out_ch) in enumerate(zip(block_counts, channels)):
            for b in range(n_blocks):
                stride = 2 if (b == 0 and stage > 0) else 1
                layers.append(ResBlock1D(in_ch, out_ch, stride))
                in_ch = out_ch
        self.body = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_ch, 1))

    def forward(self, ecg, clinical_features=None):
        x = self.stem(ecg)
        x = self.body(x)
        x = self.pool(x).squeeze(-1)
        return self.head(x).squeeze(-1), None


# ---------------------------------------------------------------
# 3. LSTM
# ---------------------------------------------------------------
class LSTMBaseline(nn.Module):
    def __init__(self, n_leads=12, hidden_dim=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(n_leads, hidden_dim, num_layers=num_layers,
                             batch_first=True, dropout=dropout, bidirectional=False)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_dim, 1))

    def forward(self, ecg, clinical_features=None):
        x = ecg.transpose(1, 2)  # (B, T, n_leads)
        out, (h_n, _) = self.lstm(x)
        last_hidden = h_n[-1]
        return self.head(last_hidden).squeeze(-1), None


# ---------------------------------------------------------------
# 4. Bi-LSTM + Attention
# ---------------------------------------------------------------
class BiLSTMAttention(nn.Module):
    def __init__(self, n_leads=12, hidden_dim=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(n_leads, hidden_dim, num_layers=num_layers,
                             batch_first=True, dropout=dropout, bidirectional=True)
        self.attn_score = nn.Linear(hidden_dim * 2, 1)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_dim * 2, 1))

    def forward(self, ecg, clinical_features=None):
        x = ecg.transpose(1, 2)  # (B, T, n_leads)
        out, _ = self.lstm(x)     # (B, T, 2*hidden_dim)
        weights = torch.softmax(self.attn_score(out), dim=1)  # (B, T, 1)
        pooled = (out * weights).sum(dim=1)                     # (B, 2*hidden_dim)
        return self.head(pooled).squeeze(-1), None


# ---------------------------------------------------------------
# 5. Transformer
# ---------------------------------------------------------------
class TransformerBaseline(nn.Module):
    """VERIFIED IMPLEMENTATION NOTE (see models/tft_branch.py's
    TFTBranch docstring for the full explanation): 4 stacked
    self-attention layers over the raw T=5000 ECG samples is not
    tractable at batch_size=256 -- each layer's attention-weight tensor
    alone would need ~205 GB. An nn.AvgPool1d downsamples T=5000 -> 200
    right after the input projection, before any attention layer, for
    the same reason and by the same factor as TFTBranch."""

    def __init__(self, n_leads=12, d_model=128, n_heads=8, num_layers=4,
                 dim_feedforward=256, dropout=0.3, attn_pool_size=25):
        super().__init__()
        self.input_proj = nn.Linear(n_leads, d_model)
        self.attn_pool = nn.AvgPool1d(kernel_size=attn_pool_size, stride=attn_pool_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(d_model, 1))

    def forward(self, ecg, clinical_features=None):
        x = ecg.transpose(1, 2)      # (B, T, n_leads)
        x = self.input_proj(x)        # (B, T, d_model)
        x = self.attn_pool(x.transpose(1, 2)).transpose(1, 2)  # (B, T', d_model)
        x = self.encoder(x)
        pooled = x.mean(dim=1)
        return self.head(pooled).squeeze(-1), None


# ---------------------------------------------------------------
# 6. TabNet-only  (== ablation A2)
# ---------------------------------------------------------------
class TabNetOnly(nn.Module):
    def __init__(self, input_dim=25, embed_dim=64, n_steps=5, n_shared=2, dropout=0.3):
        super().__init__()
        self.tabnet = TabNetBranch(input_dim=input_dim, embed_dim=embed_dim,
                                    n_steps=n_steps, n_shared=n_shared)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(embed_dim, 1))

    def forward(self, ecg, clinical_features):
        emb, mask_loss = self.tabnet(clinical_features)
        return self.head(emb).squeeze(-1), mask_loss


# ---------------------------------------------------------------
# 7. TFT-only  (== ablation A1)
# ---------------------------------------------------------------
class TFTOnly(nn.Module):
    def __init__(self, n_leads=12, seq_len=5000, embed_dim=64,
                 lstm_layers=2, attn_heads=8, dropout=0.3):
        super().__init__()
        self.tft = TFTBranch(n_leads=n_leads, seq_len=seq_len, embed_dim=embed_dim,
                              lstm_layers=lstm_layers, attn_heads=attn_heads)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(embed_dim, 1))

    def forward(self, ecg, clinical_features=None):
        emb = self.tft(ecg)
        return self.head(emb).squeeze(-1), None


BASELINE_REGISTRY = {
    "cnn_1d": CNN1D,
    "resnet34_1d": ResNet34_1D,
    "lstm": LSTMBaseline,
    "bilstm_attention": BiLSTMAttention,
    "transformer": TransformerBaseline,
    "tabnet_only": TabNetOnly,
    "tft_only": TFTOnly,
}
