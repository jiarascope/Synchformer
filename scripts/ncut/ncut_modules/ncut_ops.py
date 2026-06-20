from __future__ import annotations

import torch


def run_ncut(features: torch.Tensor, num_eig: int, device: torch.device) -> torch.Tensor:
    """
    features: [N, D]
    returns eig: [N, num_eig]-ish
    """
    try:
        from ncut_pytorch import NCUT
        eig = NCUT(num_eig=num_eig).fit_transform(features)
    except Exception:
        # Some versions use Ncut and n_eig.
        from ncut_pytorch import Ncut
        eig = Ncut(n_eig=num_eig, device=str(device)).fit_transform(features)

    if not torch.is_tensor(eig):
        eig = torch.as_tensor(eig)

    return eig.float()


# -----------------------------
# Visualization helpers
# -----------------------------
