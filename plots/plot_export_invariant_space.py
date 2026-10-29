"""
Picked distillation inputs in the invariant space: the export states (f3x3.npy of a pytorch_export_* folder) drawn on
the invariant_space_modes figure (training invariants, interpolation hull, standard mode paths).

Usage: python3 plots/plot_export_invariant_space.py --export_dir <extracted>/pytorch_export_<mode>
Writes <export_dir>/export_invariant_space_modes.pdf.
"""
import argparse
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from core.hsgp_model import HSGPHyperelasticity  # noqa: E402
from plots.training import plot_invariant_space_modes  # noqa: E402


def plot_export(export_dir, saved_model_dir=None, gp=None, label=None):
    export_dir = os.path.abspath(export_dir)
    saved_model_dir = saved_model_dir or os.path.dirname(export_dir)
    gp = gp or HSGPHyperelasticity.load(os.path.join(saved_model_dir, "hsgp_posterior.npz"))
    F = jnp.asarray(np.load(os.path.join(export_dir, "f3x3.npy")))
    dev, vol = jax.vmap(gp.feature_extractor.extract)(F)[:2]
    picked = np.concatenate([np.asarray(dev), np.asarray(vol)], axis=1)
    I_obs = np.load(os.path.join(saved_model_dir, "I_obs_all.npy"))
    label = label or "export: " + os.path.basename(export_dir).replace("pytorch_export_", "")
    plot_invariant_space_modes(gp, export_dir, I_obs, picked=picked, picked_label=label,
                               filename="export_invariant_space_modes.pdf")
    return os.path.join(export_dir, "export_invariant_space_modes.pdf")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--export_dir", required=True)
    p.add_argument("--saved_model_dir", default=None, help="extraction directory (default: parent of --export_dir)")
    args = p.parse_args()
    print(plot_export(args.export_dir, args.saved_model_dir))


if __name__ == "__main__":
    main()
