# ============================================================
#  seed_utils.py
#  Single source of truth for the fixed random seed, imported by
#  every training/evaluation script so all runs are deterministic.
#  Matches the manuscript's Section 6.2 statement: "Reproducibility
#  was ensured by fixing all random seeds (Python, NumPy, PyTorch,
#  CUDA)."
# ============================================================

import os
import random

SEED = 42  # fixed seed used throughout this repository


def fix_all_seeds(seed: int = SEED):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    import numpy as np
    np.random.seed(seed)

    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Deterministic cuDNN (slower, but required for reproducibility)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    return seed
