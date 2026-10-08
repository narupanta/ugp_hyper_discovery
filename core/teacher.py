"""
Teacher model loading for the steps after extraction (distillation export, distillation plots, validation plots).

An extraction directory holds either an SVGP model (best_params.npy + I_z.npy, rebuilt by each script as before) or an
HSGP posterior (hsgp_posterior.npz from extraction/train_hsgp.py). HSGPHyperelasticity has the same evaluation
interface (psi_dist, dev/vol_psi_dist, *_joint_cov, dev/vol_gp_mean, piola_dist, piola, get_path_psi_fn, ...), so the
scripts only need to know which one to use.
"""
import os
from typing import Optional

from .hsgp_model import HSGPHyperelasticity

HSGP_FILE = "hsgp_posterior.npz"


def is_hsgp_dir(saved_model_dir: str) -> bool:
    return os.path.exists(os.path.join(saved_model_dir, HSGP_FILE))


def load_hsgp_teacher(saved_model_dir: str) -> Optional[HSGPHyperelasticity]:
    """The HSGP posterior of an extraction directory, or None for an SVGP directory."""
    path = os.path.join(saved_model_dir, HSGP_FILE)
    if not os.path.exists(path):
        return None
    print(f"[teacher] HSGP posterior loaded from {path}")
    model = HSGPHyperelasticity.load(path)
    obs = os.path.join(saved_model_dir, "I_obs_all.npy")
    if os.path.exists(obs):   # data support used by regime plots in place of SVGP inducing points
        import numpy as np
        I_obs = np.load(obs)
        model.dev_z, model.vol_z = I_obs[:, :2], I_obs[:, 2:3]
    return model
