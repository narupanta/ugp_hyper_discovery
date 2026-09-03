import os
import sys
import json
import argparse
import numpy as np
import jax
from jax import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.model import SparseHyperelasticityGP
from core.dataclass import GPRawParams
from core.material_models import get_material_from_dir
from core.plotter import (
    plot_combined_validation,
    plot_energy_decomposition_validation,
    plot_training_r2
)
from core.features import IsotropicFeatureExtractor, AnisotropicFeatureExtractor
from core.utils import infer_material_model_name, fto3x3

def find_dataset_path(saved_model_dir, true_model_name):
    saved_dir_abs = os.path.abspath(saved_model_dir)
    all_parts = saved_dir_abs.split(os.sep)

    disp_noise = "0.0001"
    load_noise = "0.01"
    
    for p in reversed(all_parts):
        subparts = p.split('_')
        for sp in subparts:
            if sp.startswith("d") and sp[1:].replace('.', '', 1).isdigit():
                disp_noise = sp[1:]
            elif sp.startswith("l") and sp[1:].replace('.', '', 1).isdigit():
                load_noise = sp[1:]
            elif sp.replace('.', '', 1).isdigit() and sp in subparts:
                if disp_noise == "0.0001" and float(sp) < 0.005:
                    disp_noise = sp

    for search_dir in ["dataset/preprocessed/syn_f", "dataset/precomputed_vfm"]:
        if os.path.exists(search_dir):
            for fname in os.listdir(search_dir):
                if (fname.startswith(f"{true_model_name}_{disp_noise}_{load_noise}") or fname.startswith(f"{true_model_name}_")) and fname.endswith(".npz"):
                    return os.path.join(search_dir, fname)
    return None

def main():
    parser = argparse.ArgumentParser(description="Regenerate extraction plots for saved model directory.")
    parser.add_argument("--saved_model_dir", type=str, required=True, help="Path to extracted model directory")
    args = parser.parse_args()

    saved_model_dir = os.path.abspath(args.saved_model_dir)
    print(f"Loading extracted model from: {saved_model_dir}")

    # Load configuration & true model
    true_model = get_material_from_dir(saved_model_dir, jit_P=False)
    true_model_name = infer_material_model_name(saved_model_dir)
    print(f"Loaded material model: {true_model_name}")

    # Load GP parameters
    best_params_dict = np.load(os.path.join(saved_model_dir, "best_params.npy"), allow_pickle=True).item()
    best_params = GPRawParams(**best_params_dict)
    I_z = jnp.load(os.path.join(saved_model_dir, "I_z.npy"))

    dev_z = I_z[:, :2]
    if I_z.shape[1] > 3:
        vol_z = I_z[:, 2:3]
        aniso_z = I_z[:, 3:]
    elif I_z.shape[1] == 3:
        vol_z = I_z[:, 2:3]
        aniso_z = None
    else:
        vol_z = I_z[:, 2:]
        aniso_z = None

    min_dev = jnp.min(dev_z, axis=0)
    min_vol = jnp.min(vol_z, axis=0)
    max_dev = jnp.max(dev_z, axis=0)
    max_vol = jnp.max(vol_z, axis=0)
    min_aniso = jnp.min(aniso_z, axis=0) if aniso_z is not None else None
    max_aniso = jnp.max(aniso_z, axis=0) if aniso_z is not None else None

    metadata_path = os.path.join(saved_model_dir, "metadata.json")
    cov_mode = "diag"
    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            cov_mode = json.load(f).get("covariance_mode", "diag")

    feature_extractor = None
    if aniso_z is not None:
        a0_val = getattr(true_model, "a0", None)
        if a0_val is None:
            a0_val = getattr(true_model, "a1", None)
        a1_val = getattr(true_model, "a1", None) if getattr(true_model, "a0", None) is not None else getattr(true_model, "a2", None)
        if a0_val is not None and a1_val is not None:
            feature_extractor = AnisotropicFeatureExtractor(np.array(a0_val), a1=np.array(a1_val))
        elif a0_val is not None:
            feature_extractor = AnisotropicFeatureExtractor(np.array(a0_val))

    learned_gp = SparseHyperelasticityGP(
        raw_params=best_params, I_z=I_z, min_dev=min_dev, min_vol=min_vol, max_dev=max_dev, max_vol=max_vol,
        beta=1.0, feature_extractor=feature_extractor,
        min_aniso=min_aniso, max_aniso=max_aniso, aniso_z=aniso_z,
        covariance_mode=cov_mode
    )

    # 1. Combined Validation Plot (Standard loading paths)
    print("Generating combined validation plot (standard loading paths)...")
    plot_combined_validation(learned_gp, true_model, saved_model_dir, step=40000)

    # 2. Energy Decomposition Plot
    print("Generating energy decomposition plot...")
    plot_energy_decomposition_validation(learned_gp, true_model, saved_model_dir)

    # 3. Training Parity Plot
    dataset_path = find_dataset_path(saved_model_dir, true_model_name)
    if dataset_path and os.path.exists(dataset_path):
        print(f"Generating training parity plot from dataset: {dataset_path}...")
        prep_data = np.load(dataset_path, allow_pickle=True)
        F_train_full_3x3 = jax.vmap(jax.vmap(fto3x3))(prep_data["F"])
        r2, rmse, coverage = plot_training_r2(learned_gp, true_model, F_train_full_3x3, saved_model_dir)
        print(f"Training Parity Metrics: R2={r2:.4f}, RMSE={rmse:.4f}, EC={coverage:.1f}%")

    print(f"🎉 Successfully regenerated all extraction plots in: {saved_model_dir}")

if __name__ == "__main__":
    main()
