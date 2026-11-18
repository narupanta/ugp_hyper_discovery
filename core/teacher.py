"""
Teacher model loading for the steps after extraction (distillation export, distillation plots, validation plots).

An extraction directory holds the GP posterior written by pipeline/fit_gp.py (hsgp_posterior.npz): an HSGP
(core/hsgp_model.py) or an exact GP (core/exact_gp.py). Both have the same evaluation interface (psi_dist,
dev/vol_psi_dist, *_joint_cov, dev/vol_gp_mean, piola_dist, piola, get_path_psi_fn, ...).
"""
import os

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


def load_hsgp_teacher(saved_model_dir: str):
    """The GP posterior (HSGP or exact GP) of an extraction directory. The observed invariants (I_obs_all.npy) are
    attached as dev_z / vol_z: the data support of the regime (interpolation / extrapolation) plots."""
    path = os.path.join(saved_model_dir, HSGP_FILE)
    if not os.path.exists(path):
        raise FileNotFoundError(f"no GP posterior {HSGP_FILE} in {saved_model_dir} (run pipeline/fit_gp.py)")
    model = load_energy_gp(path)
    print(f"[teacher] {'exact GP' if getattr(model, 'model_type', 'hsgp') == 'exact_gp' else 'HSGP'} posterior loaded from {path}")
    obs = os.path.join(saved_model_dir, "I_obs_all.npy")
    if os.path.exists(obs):
        import numpy as np
        I_obs = np.load(obs)
        model.dev_z, model.vol_z = I_obs[:, :2], I_obs[:, 2:3]
    return model
