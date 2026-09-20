# ============================================================
#  models/tabnet_branch.py
#  Clinical branch: TabNet encoder over the 25-dim structured
#  clinical feature vector, matching Table 2:
#    N_steps=5, N_a=64 (attention/decision dim), N_shared=2
#  Uses the pytorch-tabnet library's TabNetEncoder directly, per
#  Section 6.2 ("...using PyTorch 2.1.0 and the pytorch-tabnet
#  library"), wrapped to output a fixed 64-dim embedding.
#
#  Requires: pip install pytorch-tabnet
# ============================================================

import torch
import torch.nn as nn

from pytorch_tabnet.tab_network import TabNetEncoder


class TabNetBranch(nn.Module):
    def __init__(self, input_dim=25, embed_dim=64, n_steps=5, n_shared=2,
                 gamma=1.3, dropout=0.0):
        super().__init__()
        self.encoder = TabNetEncoder(
            input_dim=input_dim,
            output_dim=embed_dim,
            n_d=embed_dim,
            n_a=embed_dim,
            n_steps=n_steps,
            gamma=gamma,
            n_independent=2,
            n_shared=n_shared,
        )
        self.embed_dim = embed_dim

    def forward(self, clinical_features):
        # clinical_features: (B, input_dim)
        steps_output, mask_loss = self.encoder(clinical_features)
        # steps_output is a list of per-step (B, n_d) tensors; sum them
        # (standard TabNet aggregation) to get the final embedding.
        embedding = torch.sum(torch.stack(steps_output, dim=0), dim=0)
        return embedding, mask_loss  # mask_loss: sparsity regularization term

    def get_attention_masks(self, clinical_features):
        """Returns the per-decision-step, feature-level attention masks
        (each of shape (B, input_dim)), as a list ordered by decision
        step, for the feature-importance analysis in Section 6.6 /
        Table 7's Eq. (6): eta_j = (1/N_steps) * sum_t M_t[j].

        NOTE: pytorch_tabnet's TabNetEncoder.forward_masks() returns
        (M_explain, masks) -- exactly 2 values, with `masks` a dict
        keyed by decision-step index -- not the 3-tuple this method
        previously (incorrectly) tried to unpack, which would have
        raised a ValueError the first time anyone actually called this
        method. `M_explain` is pytorch_tabnet's own step-importance-
        weighted aggregate (closer to what TabNetClassifier.explain()
        returns) and is intentionally NOT used here, since Eq. (6)
        specifies a plain unweighted average across steps."""
        _, masks = self.encoder.forward_masks(clinical_features)
        return [masks[step] for step in sorted(masks.keys())]
