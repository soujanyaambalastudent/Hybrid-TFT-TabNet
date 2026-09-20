# ============================================================
#  models/tft_branch.py
#  ECG branch: Temporal Fusion Transformer-style encoder, matching
#  Table 2 and the architecture description in the manuscript:
#    - Linear projection of each lead to a 64-dim embedding
#    - Sinusoidal positional encoding
#    - Variable Selection Network (softmax-normalized GRN over
#      temporal positions)
#    - 2-layer bidirectional LSTM, 64 hidden units/direction
#      (128 total), dropout 0.1
#    - 8-head self-attention, model dim 128
#    - 3-layer Gated Residual Network (GRN), ELU activation
#    - Temporal mean-pool -> per-lead 64-dim vector -> lead-mean
#      -> final 64-dim TFT embedding
# ============================================================

import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        # x: (B, T, d_model)
        return x + self.pe[:, : x.size(1)]


class GatedResidualNetwork(nn.Module):
    """3-layer GRN with ELU activation and a gating mechanism."""

    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.fc3 = nn.Linear(dim, dim)
        self.elu = nn.ELU()
        self.dropout = nn.Dropout(dropout)
        self.gate = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        h = self.elu(self.fc1(x))
        h = self.elu(self.fc2(h))
        h = self.dropout(self.fc3(h))
        g = torch.sigmoid(self.gate(x))
        return self.norm(x + g * h)


class VariableSelectionNetwork(nn.Module):
    """Softmax-normalized importance weighting over temporal positions."""

    def __init__(self, dim):
        super().__init__()
        self.grn = GatedResidualNetwork(dim)
        self.score = nn.Linear(dim, 1)

    def forward(self, x):
        # x: (B, T, dim)
        h = self.grn(x)
        weights = torch.softmax(self.score(h), dim=1)  # (B, T, 1)
        return h * weights


class TFTBranch(nn.Module):
    """Processes one 12-lead ECG record -> 64-dim embedding.

    IMPORTANT, VERIFIED IMPLEMENTATION NOTE: full self-attention over the
    raw T=5000 time steps, as a literal reading of "multi-head
    self-attention... on top of the time dimension" (Section 4.4) would
    imply, is NOT computationally tractable at the manuscript's stated
    batch size of 256 (Section 4.7) -- confirmed by actually running it:
    PyTorch's nn.MultiheadAttention materializes a (B*heads, T, T)
    attention-weight tensor, which for B=256, heads=8, T=5000 is
    256*8*5000*5000*4 bytes ~= 205 GB PER LEAD, far beyond a single
    A100's 40 GB VRAM (the hardware Section 4.7 states was used) and
    infeasible on any single accelerator available today.

    To make training actually runnable while staying as faithful as
    possible to the described design, the LSTM still runs over the FULL
    T=5000 sequence (so fine-grained, sample-level morphology is not
    lost), and only the self-attention stage operates on an
    average-pooled, downsampled version of the LSTM output (T=5000 ->
    T'=200, pool size 25 samples = 50 ms at 500 Hz). This keeps
    attention's role as originally intended -- modeling longer-range,
    inter-beat relationships (Section 4.4: "capture inter-beat
    variability") -- at a temporal granularity attention mechanisms are
    actually used for in practice, while removing the O(T^2) memory
    blow-up. The final temporal mean-pool (Table 3) is correspondingly
    computed over the T'=200 pooled positions rather than T=5000.
    """

    def __init__(self, n_leads=12, seq_len=5000, embed_dim=64,
                 lstm_layers=2, attn_heads=8, dropout=0.1, attn_pool_size=25):
        super().__init__()
        self.n_leads = n_leads
        self.embed_dim = embed_dim
        d_model = embed_dim * 2  # 128, after bidirectional LSTM concat

        if seq_len % attn_pool_size != 0:
            raise ValueError(
                f"seq_len ({seq_len}) must be divisible by attn_pool_size "
                f"({attn_pool_size}) for exact, remainder-free pooling."
            )

        self.input_proj = nn.Linear(1, embed_dim)
        self.pos_enc = PositionalEncoding(embed_dim, max_len=seq_len)
        self.vsn = VariableSelectionNetwork(embed_dim)

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=embed_dim,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        # Downsamples the LSTM output before self-attention -- see the
        # class docstring for why this is necessary for tractability.
        self.attn_pool = nn.AvgPool1d(kernel_size=attn_pool_size, stride=attn_pool_size)

        self.attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=attn_heads, batch_first=True, dropout=dropout
        )
        self.grn = GatedResidualNetwork(d_model, dropout=dropout)
        self.out_proj = nn.Linear(d_model, embed_dim)

    def _encode_one_lead(self, lead_signal):
        # lead_signal: (B, T) -> (B, T, 1) -> (B, T, embed_dim)
        x = lead_signal.unsqueeze(-1)
        x = self.input_proj(x)
        x = self.pos_enc(x)
        x = self.vsn(x)
        x, _ = self.lstm(x)                       # (B, T, 2*embed_dim), full resolution
        x = self.attn_pool(x.transpose(1, 2)).transpose(1, 2)  # (B, T', 2*embed_dim)
        attn_out, _ = self.attn(x, x, x)           # (B, T', 2*embed_dim)
        x = self.grn(attn_out)
        x = x.mean(dim=1)                          # temporal mean-pool -> (B, 2*embed_dim)
        x = self.out_proj(x)                        # -> (B, embed_dim)
        return x

    def forward(self, ecg):
        # ecg: (B, n_leads, T)
        lead_embeddings = [self._encode_one_lead(ecg[:, lead, :]) for lead in range(self.n_leads)]
        stacked = torch.stack(lead_embeddings, dim=1)  # (B, n_leads, embed_dim)
        return stacked.mean(dim=1)                       # lead-mean-pool -> (B, embed_dim)
