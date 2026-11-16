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


def load_energy_gp(path: str):
    """HSGPHyperelasticity or ExactGPHyperelasticity (core/exact_gp.py), from the model_type stored in the file."""
    import numpy as np
    with np.load(path, allow_pickle=False) as d:
        kind = str(d["model_type"]) if "model_type" in d.files else "hsgp"
    if kind == "exact_gp":
        from .exact_gp import ExactGPHyperelasticity
        return ExactGPHyperelasticity.load(path)
    return HSGPHyperelasticity.load(path)


def load_hsgp_teacher(saved_model_dir: str) -> Optional[HSGPHyperelasticity]:
    """The HSGP posterior of an extraction directory, or None for an SVGP directory."""
    path = os.path.join(saved_model_dir, HSGP_FILE)
    if not os.path.exists(path):
        return None
    model = load_energy_gp(path)
    print(f"[teacher] {'exact GP' if getattr(model, 'model_type', 'hsgp') == 'exact_gp' else 'HSGP'} posterior loaded from {path}")
    obs = os.path.join(saved_model_dir, "I_obs_all.npy")
    if os.path.exists(obs):   # data support used by regime plots in place of SVGP inducing points
        import numpy as np
        I_obs = np.load(obs)
        model.dev_z, model.vol_z = I_obs[:, :2], I_obs[:, 2:3]
    return model
