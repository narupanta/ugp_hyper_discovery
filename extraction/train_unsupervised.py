import os
import json
import yaml
import datetime
import argparse
import numpy as np
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
import jax.random as jr
import optax

# Enforce mandatory 64-bit precision standard
jax.config.update("jax_enable_x64", True)

from core.model import SparseHyperelasticityGP
from core.utils import fto3x3, farthest_point_sampling_with_fixed_point, load_f3x3_from_dataset
from core.dataclass import GPRawParams
from core.material_models import get_material
from core.trainer import HyperelasticGPTrainer
from core.features import IsotropicFeatureExtractor, AnisotropicFeatureExtractor
from core.datasetclass import DatasetFactory
from core.dataset_store import dataset_exists
from core.loss_function import (total_stochastic_loss, build_eiv_indices, eiv_force_equivalent_sigma,
                                eiv_linearisation, internal_force, eiv_noise_estimate, damped_newton_step,
                                ellipticity_penalty)
from core.fem_engine import make_plane_stress_piola
from core.plotter import (
    plot_inducing_points, plot_training_r2,
    plot_domain_invariants, plot_reaction_forces_noise_comparison
)

def parse_args():
    parser = argparse.ArgumentParser(description="Isihara Model Dataset and Training Configuration")

    # Dataset & Model Config
    parser.add_argument('--recipe', type=str, default=None, help="Path to recipe YAML configuration file")
    parser.add_argument('--material_model_name', type=str, default="isihara")
    parser.add_argument('--disp_noise', type=float, default=0.0001)
    parser.add_argument('--load_noise', type=float, default=0.01)
    parser.add_argument('--target_load_true_top', type=float, default=8.0)
    parser.add_argument('--asym_factor', type=float, default=0.95)
    parser.add_argument('--model_mode', type=str, default='isotropic')
    parser.add_argument('--dataset_path', type=str, default="", help="Direct path to precomputed dataset npz file. Overrides heuristic path search.")
    parser.add_argument('--sampling_mode', type=str, default='pathwise', choices=['pathwise', 'cholesky', 'pws', 'mds'],
                        help="Sampling mode for GP realizations: 'pathwise' (RFF + Matheron's rule) or 'cholesky' (multivariate normal)")

    # Training Config
    parser.add_argument('--number_of_mci_sampling', type=int, default=3)
    parser.add_argument('--n_ip', type=int, default=5)
    parser.add_argument('--beta', type=float, default=50.0)
    parser.add_argument('--num_rff', type=int, default=200, help="Number of Random Fourier Features basis")
    
    # Booleans (using 0/1 as integers is often safer in shell scripts)
    parser.add_argument('--is_fixed_reaction_force_noise', type=int, default=1)
    parser.add_argument('--is_fixed_inducing_points', type=int, default=1, help="Set to 1 to freeze inducing points at FPS picked locations, 0 to optimize them")
    parser.add_argument('--cap_compression', type=int, default=1, help="Set to 1 to cap anisotropic invariants to >= 0 (no compression stiffness)")

    # Handling the List [1, 5, 9] to cover the 10 steps range
    parser.add_argument('--train_load_steps_indices', type=int, nargs='+', default=[1, 5, 9])
    parser.add_argument('--val_load_steps_indices', type=int, nargs='+', default=None, help="Validation load steps indices for R2 evaluation")
    parser.add_argument('--test_load_steps_indices', type=int, nargs='+', default=None, help="Test load steps indices for extrapolation evaluation")
    parser.add_argument('--n_iterations', type=int, default=1000)
    parser.add_argument('--learning_rate', type=float, default=0.01, help="Learning rate for Adam optimizer")
    parser.add_argument('--final_learning_rate', type=float, default=None, help="Final learning rate for cosine decay. If not set or equal to learning_rate, uses constant lr.")
    parser.add_argument('--geometry', type=str, default='block', help="Geometry of the specimen")
    
    # Resume training
    parser.add_argument('--resume_from', type=str, default="", help="Name of the extraction/extracted_models folder to resume from")
    
    parser.add_argument('--seed', type=int, default=42, help="Random seed for PRNGKey")
    parser.add_argument('--batch_dir', type=str, default="", help="If provided, models are saved into batch_dir/seed")
    parser.add_argument("--covariance_mode", type=str, default="diag", choices=["diag", "full", "whitened_diag", "whitened_full"], help="Covariance matrix parameterization for inducing points.")
    parser.add_argument('--angles', type=float, nargs='+', default=None)
    parser.add_argument('--dev_params', type=float, nargs='+', default=None)
    parser.add_argument('--vol_params', type=float, nargs='+', default=None)
    parser.add_argument('--aniso_params', type=float, nargs='+', default=None)
    parser.add_argument('--normalize_ell', type=int, default=0, choices=[0, 1], help="Whether to normalize expected log-likelihood by degrees of freedom to prevent uncertainty collapse (1) or use unnormalized sum (0)")
    parser.add_argument('--u_var_anchor', type=float, default=1e-12, help="Anchor point variance (default 1e-12)")
    parser.add_argument('--kzz_jitter', type=float, default=1e-8, help="Kzz diagonal value: fixed numerical jitter (default), or initial value when --trainable_kzz_noise is set")
    parser.add_argument('--trainable_kzz_noise', action='store_true', default=False, help="Make Kzz diagonal noise a trainable parameter (learned alongside sigma_free)")
    parser.add_argument('--vfm_mode', type=str, default="linear_triangle",
                        choices=["linear_triangle", "global_vf", "mix"],
                        help="VFM loss mode: 'linear_triangle', 'global_vf', or 'mix'")
    parser.add_argument('--vf_order', type=int, default=2,
                        help="Polynomial order for kinematically admissible virtual fields basis (default: 2)")
    parser.add_argument('--stress_mode', type=str, default=None, choices=["plane_strain", "plane_stress"],
                        help="Stress state assumption: 'plane_strain' or 'plane_stress'. If None, loaded from recipe or dataset.")
    parser.add_argument('--control_mode', type=str, default=None, choices=["force", "displacement"],
                        help="Control mode: 'force' or 'displacement'. If None, loaded from recipe or dataset.")
    parser.add_argument('--constraint_lengthscale', type=int, default=None, choices=[0, 1],
                        help="Whether to constrain lengthscales to domain bounds (1) or unconstrained softplus (0, allows ARD pruning). If None, loaded from recipe or defaults to 1.")
    parser.add_argument('--reaction_loss_weight', type=str, default=None,
                        help="Scaling coefficient for reaction loss (fix_x, fix_y). Can be a float (e.g. '1.0'), or 'auto' / 'ratio' to scale by #free_nodes / #boundary_nodes. If None, loaded from recipe or defaults to 1.0.")
    parser.add_argument('--free_noise_mode', type=str, default="constant",
                        choices=["constant", "nodal", "diagonal"],
                        help="Noise variance parameterization for free PDE nodes: 'constant' (single scalar for all free nodes) or 'nodal'/'diagonal' (independent per-node variance).")
    parser.add_argument('--likelihood', type=str, default=None, choices=["auto", "residual", "eiv", "residual_then_eiv"],
                        help="'residual': iid Gaussian nodal force residuals. 'eiv': errors-in-variables likelihood in displacement space "
                             "(Gauss-Newton step eps = K^-1 r; scale-invariant, so no reaction re-weighting is needed). "
                             "'auto' (default) picks 'eiv' for displacement control with linear_triangle VFM. If None, loaded from recipe.")
    parser.add_argument('--eiv_switch_fraction', type=float, default=None,
                        help="residual_then_eiv: fraction of n_iterations trained with the residual likelihood before switching "
                             "to EIV (robust far from the solution; EIV is the calibrated likelihood near it). Recipe key or 0.5.")
    parser.add_argument('--eiv_switch_mode', type=str, default=None, choices=["fraction", "plateau"],
                        help="residual_then_eiv: switch at a fixed fraction, or when the stage-1 ELBO plateaus "
                             "(window-mean improvement < plateau_rel_tol*|loss| for plateau_patience windows). Recipe key or 'fraction'.")
    parser.add_argument('--eiv_switch_min_fraction', type=float, default=None, help="plateau mode: earliest switch (fraction of n_iterations). Recipe key or 0.2.")
    parser.add_argument('--eiv_switch_max_fraction', type=float, default=None, help="plateau mode: latest switch (fraction of n_iterations). Recipe key or 0.7.")
    parser.add_argument('--plateau_window', type=int, default=None, help="plateau mode: window length in iterations. Recipe key or 5000.")
    parser.add_argument('--plateau_rel_tol', type=float, default=None, help="plateau mode: relative improvement threshold per window. Recipe key or 1e-3.")
    parser.add_argument('--plateau_patience', type=int, default=None, help="plateau mode: consecutive flat windows needed. Recipe key or 2.")
    parser.add_argument('--n_restarts', type=int, default=None,
                        help="Number of random GP initialisations trained briefly with the stage-1 objective; training continues "
                             "from the lowest-loss one. Recipe key or 1 (no restarts).")
    parser.add_argument('--restart_iterations', type=int, default=None, help="Iterations per restart (extra to n_iterations). Recipe key or 5000.")
    parser.add_argument('--stability_prior_weight', type=float, default=None,
                        help="Weight w of the strong-ellipticity prior on the posterior-mean energy at the training states: "
                             "w * sum relu(-lambda_min(Q)/s_ref)^2 over states and directions (zero for stable materials). "
                             "0 disables. Recipe key or 0.")
    parser.add_argument('--stability_directions', type=int, default=None, help="Directions n per state for the acoustic tensor. Recipe key or 8.")
    parser.add_argument('--prior_mean', type=str, default=None, choices=["none", "linear_elastic"],
                        help="'linear_elastic': explicit basis psi += mu (I1_bar-3)/2 + kappa (J-1)^2/2 with a Gaussian posterior over "
                             "(mu, kappa) (full 2x2 covariance) and a broad zero-mean prior; the GPs model the deviations. Recipe key or 'none'.")
    parser.add_argument('--linear_elastic_prior_scale', type=float, default=None,
                        help="Prior std of (mu, kappa). Recipe key or 100 x data energy density.")
    parser.add_argument('--hyperparameter_init', type=str, default=None, choices=["random", "data"],
                        help="GP hyperparameter starts: 'random' (raw ~ N(0,1)) or 'data' (lengthscale = feature span, amplitude = "
                             "factor * data energy density). Inducing values stay random. Recipe key or 'random'.")
    parser.add_argument('--hyperparameter_amplitude_factor', type=float, default=None,
                        help="hyperparameter_init=data: initial GP amplitude as a multiple of the data energy density. Recipe key or 10.")
    parser.add_argument('--hyperparameter_lengthscale_factor', type=float, default=None,
                        help="hyperparameter_init=data: initial lengthscale as a multiple of the inducing-feature span (< 2 when constrained). Recipe key or 1.5.")
    parser.add_argument('--amplitude_prior_scale', type=float, default=None,
                        help="Log-normal hyperprior on the GP amplitudes: log sig ~ N(log(amplitude_factor * data energy density), tau^2), "
                             "tau = this value (MAP-II instead of ML-II). Stops a GP from collapsing to sig -> 0, an absorbing state under "
                             "whitening (dL/dv ~ sig, dL/dsig ~ v). 0 disables. Recipe key or 0.")
    parser.add_argument('--freeze_amplitudes_stage1', type=str, default=None,
                        help="residual_then_eiv: hold the GP amplitudes at their initial values during the residual stage "
                             "(true/false). Recipe key or false.")
    parser.add_argument('--eiv_learning_rate', type=float, default=None,
                        help="residual_then_eiv: initial learning rate of the fresh Adam in the EIV stage (cosine-decayed to 10%%). Recipe key or 1e-3.")
    parser.add_argument('--eiv_damping', type=float, default=None,
                        help="Tikhonov damping of the EIV Newton step, relative to the rms singular value of the tangent "
                             "(0 = exact step). Bounds the step where the mean energy is locally unstable. Recipe key or 0.")
    parser.add_argument('--noise_prior_dof', type=float, default=None,
                        help="Degrees of freedom nu of the hierarchical prior sigma_i^2 ~ InvGamma(nu/2, nu/2 * sigma_global^2) on per-node "
                             "noise (nodal/diagonal free_noise_mode); larger = stronger pooling towards sigma_global. Integrated out under "
                             "'eiv', MAP penalty under 'residual' (<= 0 disables there). If None, loaded from recipe or defaults to 4.")
    return parser.parse_args()

def sigma_fix_to_log_sigma_fix(sigma_fix) :
    return jnp.log(jnp.maximum(sigma_fix, 1e-3))

def inv_softplus(y):
    """Computes initial raw parameters from physical coordinates in invariant space."""
    y_safe = jnp.maximum(y, 1e-15)
    return jnp.where(y_safe > 20.0, y_safe, jnp.log(jnp.expm1(y_safe)))

def external_work_density(prep_data, node_type, control_mode, last_step):
    """
    Average strain-energy density implied by the measurements alone: external work done up to `last_step`
    (trapezoidal rule over all load steps) divided by the specimen area. Displacement control: measured reactions
    times the mean prescribed displacement of the loaded edges; force control: Neumann nodal forces times
    the observed displacements. Falls back to 1.0 if the estimate is not positive.
    """
    u = np.asarray(prep_data["u_obs"] if "u_obs" in prep_data else prep_data["u"])[: last_step + 1]
    area = float(np.sum(np.asarray(prep_data["dA"])))
    if control_mode == "displacement":
        R = np.asarray(prep_data["reaction_forces"])[: last_step + 1]
        ux = u[:, np.asarray(node_type)[:, 3] == 1, 0].mean(axis=1) if np.any(np.asarray(node_type)[:, 3] == 1) else np.zeros(len(u))
        uy = u[:, np.asarray(node_type)[:, 4] == 1, 1].mean(axis=1) if np.any(np.asarray(node_type)[:, 4] == 1) else np.zeros(len(u))
        dW = 0.5 * (R[1:, 0] + R[:-1, 0]) * np.diff(ux) + 0.5 * (R[1:, 1] + R[:-1, 1]) * np.diff(uy)
    else:
        f = np.asarray(prep_data["f_neu"])[: last_step + 1]
        dW = np.sum(0.5 * (f[1:] + f[:-1]) * np.diff(u, axis=0), axis=(1, 2))
    e = float(np.sum(dW)) / area
    if not np.isfinite(e) or e <= 0:
        print(f"[CONFIGURATION] Warning: external-work energy estimate {e} not positive; using 1.0.")
        return 1.0
    return e


def get_freeze_fn(is_fixed_noise: bool, is_fixed_z: bool, covariance_mode: str = "diag", is_free_x=None, is_free_y=None):
    def freeze_fn(grads):
        replace_kwargs = {}

        # Anchor index 0 (reference stress-free state)
        if "full" in covariance_mode:
            raw_dev_u_var = grads.raw_dev_u_var.at[0, :].set(0.0).at[:, 0].set(0.0)
            raw_vol_u_var = grads.raw_vol_u_var.at[0, :].set(0.0).at[:, 0].set(0.0)
        else:
            raw_dev_u_var = grads.raw_dev_u_var.at[0].set(0.0)
            raw_vol_u_var = grads.raw_vol_u_var.at[0].set(0.0)

        replace_kwargs.update({
            "raw_dev_z": grads.raw_dev_z.at[0].set(0.0),
            "raw_vol_z": grads.raw_vol_z.at[0].set(0.0),
            "raw_dev_u_mean": grads.raw_dev_u_mean.at[0].set(0.0),
            "raw_dev_u_var": raw_dev_u_var,
            "raw_vol_u_mean": grads.raw_vol_u_mean.at[0].set(0.0),
            "raw_vol_u_var": raw_vol_u_var
        })

        if getattr(grads, "raw_aniso_z", None) is not None:
            if "full" in covariance_mode:
                raw_aniso_u_var = grads.raw_aniso_u_var.at[0, :].set(0.0).at[:, 0].set(0.0)
            else:
                raw_aniso_u_var = grads.raw_aniso_u_var.at[0].set(0.0)
            replace_kwargs.update({
                "raw_aniso_z": grads.raw_aniso_z.at[0].set(0.0),
                "raw_aniso_u_mean": grads.raw_aniso_u_mean.at[0].set(0.0),
                "raw_aniso_u_var": raw_aniso_u_var
            })

        # Freeze fixed node noise components if log_sigma_free is a vector (nodal/diagonal mode)
        if is_free_x is not None and getattr(grads, "log_sigma_free_x", None) is not None and grads.log_sigma_free_x.ndim > 0:
            replace_kwargs["log_sigma_free_x"] = jnp.where(is_free_x, grads.log_sigma_free_x, 0.0)
        if is_free_y is not None and getattr(grads, "log_sigma_free_y", None) is not None and grads.log_sigma_free_y.ndim > 0:
            replace_kwargs["log_sigma_free_y"] = jnp.where(is_free_y, grads.log_sigma_free_y, 0.0)

        if replace_kwargs:
            grads = grads._replace(**replace_kwargs)
        
        # 2. Optionally freeze reaction force noise parameters
        if is_fixed_noise:
            grads = grads._replace(
                log_sigma_fix_x=jnp.zeros_like(grads.log_sigma_fix_x),
                log_sigma_fix_y=jnp.zeros_like(grads.log_sigma_fix_y)
            )
            
        # 3. Optionally freeze ALL inducing point positions (from FPS)
        if is_fixed_z:
            replace_kwargs_z = {
                "raw_dev_z": jnp.zeros_like(grads.raw_dev_z),
                "raw_vol_z": jnp.zeros_like(grads.raw_vol_z)
            }
            if getattr(grads, "raw_aniso_z", None) is not None:
                # If fiber angle is known (fixed), freeze aniso inducing points as well.
                # Only keep aniso inducing points unfrozen if fiber angle is being learned dynamically.
                is_unknown_fiber = getattr(grads, "raw_aniso_theta_mean", None) is not None
                if not is_unknown_fiber:
                    replace_kwargs_z["raw_aniso_z"] = jnp.zeros_like(grads.raw_aniso_z)
            grads = grads._replace(**replace_kwargs_z)
        return grads
    return freeze_fn



if __name__ == "__main__" :
    base_save_path = "extraction/extracted_models"  # change as needed
    os.makedirs(base_save_path, exist_ok=True)
    # training_mode = "stochastic"
    args = parse_args()

    # Now use args.variable_name instead of hardcoded values
    material_model_name = args.material_model_name

    disp_noise = args.disp_noise
    load_noise = args.load_noise
    target_load_true_top = args.target_load_true_top
    asym_factor = args.asym_factor
    model_mode = args.model_mode
    number_of_mci_sampling = args.number_of_mci_sampling
    train_load_steps_indices = args.train_load_steps_indices
    n_ip = args.n_ip
    beta = args.beta
    is_fixed_reaction_force_noise = args.is_fixed_reaction_force_noise
    is_fixed_inducing_points = args.is_fixed_inducing_points

    n_iterations = args.n_iterations
    learning_rate = args.learning_rate

    timestamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    training_config_str = f"{material_model_name}_{disp_noise}_{load_noise}_{target_load_true_top}_{asym_factor}_{n_ip}_{beta}_{is_fixed_reaction_force_noise}_fip{is_fixed_inducing_points}_{model_mode}_{args.geometry}"
    
    base_save_path = "extraction/extracted_models"
    config_folder = f"{timestamp}_{material_model_name}_{disp_noise}_{load_noise}_{target_load_true_top}_{asym_factor}_{n_ip}_{beta}_{is_fixed_reaction_force_noise}_fip{is_fixed_inducing_points}_{model_mode}_{args.geometry}_{args.seed}"
    if args.batch_dir:
        save_path = os.path.abspath(args.batch_dir)
    else:
        save_path = os.path.abspath(os.path.join(base_save_path, config_folder))
        
    os.makedirs(save_path, exist_ok=True)

    config_dict = vars(args)
    with open(os.path.join(save_path, "config.json"), "w") as f:
        json.dump(config_dict, f, indent=4)
    with open(os.path.join(save_path, "config.yaml"), "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False)

    # Load defaults from recipe if available
    recipe_path = args.recipe or os.path.join("configs", "recipes", f"{material_model_name}.yaml")
    rec = {}
    if os.path.exists(recipe_path):
        with open(recipe_path, "r") as f:
            rec = yaml.safe_load(f) or {}
        if "material_model_name" in rec and args.material_model_name == "isihara" and args.recipe is not None:
            material_model_name = rec["material_model_name"]

    # load precomputed dataset
    if args.dataset_path and dataset_exists(args.dataset_path):
        prep_dataset_path = args.dataset_path
        print(f"[DATASET] Loading explicitly specified dataset: {prep_dataset_path}")
    elif args.dataset_path:
        raise FileNotFoundError(f"Explicitly specified --dataset_path not found: {args.dataset_path}")
    else:
        data_dir = "dataset/preprocessed/syn_f" if os.path.exists("dataset/preprocessed/syn_f") else "dataset/precomputed_vfm" 
        seeded_path = os.path.join(data_dir, f"{material_model_name}_{disp_noise}_{load_noise}_{target_load_true_top}_{asym_factor}_{args.geometry}_{args.seed}.npz")
        unseeded_path = os.path.join(data_dir, f"{material_model_name}_{disp_noise}_{load_noise}_{target_load_true_top}_{asym_factor}_{args.geometry}.npz")
        no_geom_path = os.path.join(data_dir, f"{material_model_name}_{disp_noise}_{load_noise}_{target_load_true_top}_{asym_factor}.npz")

        if os.path.exists(seeded_path):
            prep_dataset_path = seeded_path
            print(f"[DATASET] Loading exact seeded dataset: {prep_dataset_path}")
        elif os.path.exists(unseeded_path):
            prep_dataset_path = unseeded_path
            print(f"[DATASET] Warning: Seeded dataset not found ({seeded_path}). Loading unseeded fallback: {prep_dataset_path}")
        elif os.path.exists(no_geom_path):
            prep_dataset_path = no_geom_path
            print(f"[DATASET] Warning: Geometry dataset not found ({unseeded_path}). Loading fallback: {prep_dataset_path}")
        else:
            raise FileNotFoundError(
                f"Could not find a valid precomputed dataset for {material_model_name} (seed={args.seed}, geom={args.geometry}).\n"
                f"Searched paths:\n  - {seeded_path}\n  - {unseeded_path}\n  - {no_geom_path}\n"
                f"Please pass --dataset_path directly or generate data first."
            )
    
    dataset = DatasetFactory.create("dataset/precomputed_vfm", data_path=prep_dataset_path)
    prep_data = dataset.get_data()

    # Resolve control_mode and stress_mode
    dataset_control = prep_data.get("control_mode", None)
    if hasattr(dataset_control, "item"):
        dataset_control = dataset_control.item()
    control_mode = (args.control_mode or dataset_control or rec.get("control_mode", "force"))
    control_mode = str(control_mode).lower()

    dataset_stress = prep_data.get("stress_mode", None)
    if hasattr(dataset_stress, "item"):
        dataset_stress = dataset_stress.item()
    stress_mode = (args.stress_mode or dataset_stress or rec.get("stress_mode", "plane_strain"))
    stress_mode = str(stress_mode).lower()

    # Dataset filenames do not encode these modes, so a stale file can carry different kinematics/BCs than requested
    for name, requested, stored in [("control_mode", control_mode, dataset_control), ("stress_mode", stress_mode, dataset_stress)]:
        if stored is not None and str(stored).lower() != requested:
            raise ValueError(f"Requested {name}='{requested}' but dataset {prep_dataset_path} was generated with "
                             f"{name}='{stored}'. Regenerate the dataset or pass the matching {name}.")

    # Resolve constraint_lengthscale (default: 1 for backward compatibility)
    constraint_lengthscale = args.constraint_lengthscale
    if constraint_lengthscale is None:
        rec_cl = rec.get("constraint_lengthscale", None)
        if rec_cl is not None:
            constraint_lengthscale = int(rec_cl)
        else:
            constraint_lengthscale = 1
    else:
        constraint_lengthscale = int(constraint_lengthscale)

    f2x2 = prep_data["F"][train_load_steps_indices]
    cells = prep_data["cells"]
    node_type = np.asarray(prep_data["node_type"])
    mesh_pos = np.asarray(prep_data["mesh_pos"])
    print(f"[DATASET] Dataset successfully loaded: {cells.shape[0]} elements, {node_type.shape[0]} nodes, {prep_data['F'].shape[0]} total load steps.") 

    # Resolve reaction_loss_weight (#free_nodes / #boundary_nodes or custom scalar)
    rlw_input = args.reaction_loss_weight if args.reaction_loss_weight is not None else rec.get("reaction_loss_weight", rec.get("reaction_loss_scale", 1.0))
    if str(rlw_input).lower() in ["auto", "ratio", "scale", "node_ratio", "true"]:
        # Compute ratio: #free_nodes / #boundary_nodes
        is_fix_x = (node_type[:, 1] == 1)
        is_fix_y = (node_type[:, 2] == 1)
        is_loaded_x = (node_type[:, 3] == 1)
        is_loaded_y = (node_type[:, 4] == 1)
        if control_mode == "displacement":
            is_boundary = (is_fix_x | is_fix_y | is_loaded_x | is_loaded_y)
        else:
            is_boundary = (is_fix_x | is_fix_y | (node_type[:, 3] == 1) | (node_type[:, 4] == 1))
        n_boundary = int(np.sum(is_boundary))
        n_free = int(np.sum(~is_boundary))
        reaction_loss_weight = float(n_free) / float(max(n_boundary, 1))
        print(f"[CONFIGURATION] Auto reaction loss scaling enabled: #free_nodes={n_free}, #boundary_nodes={n_boundary} -> coeff = {reaction_loss_weight:.4f}")
    else:
        try:
            reaction_loss_weight = float(rlw_input)
        except (ValueError, TypeError):
            reaction_loss_weight = 1.0
        print(f"[CONFIGURATION] Reaction loss weight coefficient: {reaction_loss_weight}")

    # Resolve trainable_kzz_noise and kzz_jitter from CLI or recipe
    rec_trainable_kzz = rec.get("trainable_kzz_noise", False)
    if isinstance(rec_trainable_kzz, str):
        rec_trainable_kzz = rec_trainable_kzz.lower() in ["true", "1", "yes"]
    trainable_kzz_noise = bool(args.trainable_kzz_noise or rec_trainable_kzz)

    rec_kzz_jitter = rec.get("kzz_jitter", None)
    if rec_kzz_jitter is not None and args.kzz_jitter == 1e-8:
        kzz_jitter = float(rec_kzz_jitter)
    else:
        kzz_jitter = float(args.kzz_jitter)

    if trainable_kzz_noise:
        print(f"[CONFIGURATION] Trainable kzz_noise enabled (initial value: {kzz_jitter:.2e})")
    else:
        print(f"[CONFIGURATION] Fixed kzz_jitter: {kzz_jitter:.2e}")

    # Resolve free_noise_mode from CLI or recipe
    rec_free_noise_mode = rec.get("free_noise_mode", "constant")
    free_noise_mode = args.free_noise_mode if args.free_noise_mode != "constant" else rec_free_noise_mode
    free_noise_mode = str(free_noise_mode).lower()

    # Resolve likelihood: errors-in-variables (displacement space) is the default for displacement control
    likelihood = str(args.likelihood or rec.get("likelihood", "auto")).lower()
    if likelihood == "auto":
        likelihood = "eiv" if (control_mode == "displacement" and args.vfm_mode == "linear_triangle") else "residual"
    # 'residual_then_eiv': residual likelihood far from the solution (no singular-tangent barriers), EIV near it
    two_stage = likelihood == "residual_then_eiv"
    final_likelihood = "eiv" if likelihood in ("eiv", "residual_then_eiv") else "residual"

    def _rec_float(arg_val, key, default):
        return float(arg_val) if arg_val is not None else float(rec.get(key, default))
    eiv_switch_fraction = _rec_float(args.eiv_switch_fraction, "eiv_switch_fraction", 0.5)
    eiv_learning_rate = _rec_float(args.eiv_learning_rate, "eiv_learning_rate", 1e-3)
    eiv_damping = _rec_float(args.eiv_damping, "eiv_damping", 0.0)
    eiv_switch_mode = str(args.eiv_switch_mode or rec.get("eiv_switch_mode", "fraction")).lower()
    plateau_window = int(_rec_float(args.plateau_window, "plateau_window", 5000))
    plateau_rel_tol = _rec_float(args.plateau_rel_tol, "plateau_rel_tol", 1e-3)
    plateau_patience = int(_rec_float(args.plateau_patience, "plateau_patience", 2))
    eiv_switch_min_iteration = int(round(_rec_float(args.eiv_switch_min_fraction, "eiv_switch_min_fraction", 0.2) * n_iterations))
    eiv_switch_max_iteration = int(round(_rec_float(args.eiv_switch_max_fraction, "eiv_switch_max_fraction", 0.7) * n_iterations))
    n_restarts = int(_rec_float(args.n_restarts, "n_restarts", 1))
    stability_prior_weight = _rec_float(args.stability_prior_weight, "stability_prior_weight", 0.0)
    stability_directions = int(_rec_float(args.stability_directions, "stability_directions", 8))
    if stability_prior_weight > 0:
        print(f"[CONFIGURATION] Strong-ellipticity prior: weight {stability_prior_weight}, {stability_directions} directions per state.")
    restart_iterations = int(_rec_float(args.restart_iterations, "restart_iterations", 5000))
    if two_stage and eiv_switch_mode == "plateau":
        eiv_switch_iteration = None
        if not (0 < eiv_switch_min_iteration <= eiv_switch_max_iteration < n_iterations):
            raise ValueError("plateau switch needs 0 < eiv_switch_min_fraction <= eiv_switch_max_fraction < 1.")
    else:
        eiv_switch_iteration = int(round(eiv_switch_fraction * n_iterations)) if two_stage else None
        if two_stage and not (0 < eiv_switch_iteration < n_iterations):
            raise ValueError(f"eiv_switch_fraction={eiv_switch_fraction} must leave iterations for both stages.")

    if final_likelihood == "eiv":
        if control_mode != "displacement":
            raise ValueError("likelihood='eiv' is implemented for displacement control only.")
        if args.vfm_mode != "linear_triangle":
            raise ValueError("likelihood='eiv' requires vfm_mode='linear_triangle'.")
        if args.normalize_ell == 1:
            raise ValueError("likelihood='eiv' requires normalize_ell=0 (normalisation breaks the ELBO).")
        if args.sampling_mode not in ("pathwise", "pws"):
            raise ValueError("likelihood='eiv' requires pathwise sampling (it differentiates each GP path twice).")
        if reaction_loss_weight != 1.0 and not two_stage:
            print(f"[CONFIGURATION] Warning: reaction_loss_weight={reaction_loss_weight} tempers the EIV likelihood; "
                  f"the EIV free-DOF term is scale-invariant, so 1.0 is the principled value.")

    noise_prior_dof = args.noise_prior_dof if args.noise_prior_dof is not None else float(rec.get("noise_prior_dof", 4.0))
    if final_likelihood == "eiv" and free_noise_mode in ["nodal", "diagonal"] and noise_prior_dof <= 0:
        raise ValueError("likelihood='eiv' with per-node noise needs noise_prior_dof > 0 (each DOF has only a few load steps).")
    print(f"[CONFIGURATION] Likelihood: '{likelihood}' | per-node noise prior dof: {noise_prior_dof}")
    if two_stage and eiv_switch_mode == "plateau":
        print(f"[CONFIGURATION] Two-stage: residual (reaction weight {reaction_loss_weight}) until its ELBO plateaus "
              f"(window {plateau_window}, rel tol {plateau_rel_tol}, patience {plateau_patience}; between iterations "
              f"{eiv_switch_min_iteration} and {eiv_switch_max_iteration}), then EIV (damping {eiv_damping}, lr {eiv_learning_rate}).")
    elif two_stage:
        print(f"[CONFIGURATION] Two-stage: residual (reaction weight {reaction_loss_weight}) for {eiv_switch_iteration} iterations, "
              f"then EIV (weight 1, damping {eiv_damping}, lr {eiv_learning_rate}) for {n_iterations - eiv_switch_iteration}.")
    if n_restarts > 1:
        print(f"[CONFIGURATION] {n_restarts} random initialisations x {restart_iterations} iterations; continuing from the best.")

    # Identify free nodes for boundary freezing and noise initialization
    is_fix_x = (node_type[:, 1] == 1)
    is_fix_y = (node_type[:, 2] == 1)
    if control_mode == "displacement":
        is_free_x = ~(is_fix_x | (node_type[:, 3] == 1))
        is_free_y = ~(is_fix_y | (node_type[:, 4] == 1))
    else:
        is_free_x = ~is_fix_x
        is_free_y = ~is_fix_y

    is_free_x_jnp = jnp.asarray(is_free_x)
    is_free_y_jnp = jnp.asarray(is_free_y)

    # Free noise is a force-residual std ('residual') or a displacement std ('eiv'). Under 'eiv' the noise is
    # integrated out during training (only sigma_global is learned, as the per-node prior scale) and written back
    # after training; it starts at 0.1% of the RMS training displacement, a data-derived scale.
    if likelihood == "eiv":
        u_obs_train = np.asarray(prep_data["u_obs"] if "u_obs" in prep_data else prep_data["u"])[train_load_steps_indices]
        log_sigma0 = float(np.log(1e-3 * np.sqrt(np.mean(u_obs_train**2))))
    else:
        log_sigma0 = 0.0
    if free_noise_mode in ["nodal", "diagonal"]:
        n_nodes = node_type.shape[0]
        log_sigma_free_x_init = jnp.full((n_nodes,), log_sigma0, dtype=jnp.float64)
        log_sigma_free_y_init = jnp.full((n_nodes,), log_sigma0, dtype=jnp.float64)
        print(f"[CONFIGURATION] Heteroscedastic free residual noise enabled: {free_noise_mode.upper()} ({n_nodes} nodes)")
    else:
        log_sigma_free_x_init = jnp.array(log_sigma0, dtype=jnp.float64)
        log_sigma_free_y_init = jnp.array(log_sigma0, dtype=jnp.float64)
        print(f"[CONFIGURATION] Constant free residual noise scalar enabled.")

    config_dict["control_mode"] = control_mode
    config_dict["stress_mode"] = stress_mode
    config_dict["constraint_lengthscale"] = constraint_lengthscale
    config_dict["reaction_loss_weight"] = reaction_loss_weight
    config_dict["trainable_kzz_noise"] = trainable_kzz_noise
    config_dict["kzz_jitter"] = kzz_jitter
    config_dict["free_noise_mode"] = free_noise_mode
    config_dict["likelihood"] = likelihood
    config_dict["noise_prior_dof"] = noise_prior_dof
    config_dict["final_likelihood"] = final_likelihood
    config_dict["eiv_switch_iteration"] = eiv_switch_iteration
    config_dict["eiv_switch_mode"] = eiv_switch_mode
    config_dict["eiv_switch_min_iteration"] = eiv_switch_min_iteration
    config_dict["eiv_switch_max_iteration"] = eiv_switch_max_iteration
    config_dict["plateau_window"] = plateau_window
    config_dict["plateau_rel_tol"] = plateau_rel_tol
    config_dict["plateau_patience"] = plateau_patience
    config_dict["n_restarts"] = n_restarts
    config_dict["stability_prior_weight"] = stability_prior_weight
    config_dict["stability_directions"] = stability_directions
    config_dict["restart_iterations"] = restart_iterations
    config_dict["eiv_learning_rate"] = eiv_learning_rate
    config_dict["eiv_damping"] = eiv_damping
    with open(os.path.join(save_path, "config.json"), "w") as f:
        json.dump(config_dict, f, indent=4)
    with open(os.path.join(save_path, "config.yaml"), "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False)

    print(f"[CONFIGURATION] Active Control Mode: '{control_mode}', Stress State: '{stress_mode}'.") 

    mat_kwargs = {}
    if args.angles is not None: mat_kwargs["angles"] = args.angles
    if args.dev_params is not None: mat_kwargs["dev_params"] = args.dev_params
    if args.vol_params is not None: mat_kwargs["vol_params"] = args.vol_params
    if args.aniso_params is not None: mat_kwargs["aniso_params"] = args.aniso_params

    mat_p = rec.get("material_params", {})
    if "dev_params" not in mat_kwargs and "dev_params" in mat_p:
        mat_kwargs["dev_params"] = mat_p["dev_params"]
    if "vol_params" not in mat_kwargs and "vol_params" in mat_p:
        mat_kwargs["vol_params"] = mat_p["vol_params"]
    if "aniso_params" not in mat_kwargs and "aniso_params" in mat_p:
        mat_kwargs["aniso_params"] = mat_p["aniso_params"]
    if "angles" not in mat_kwargs and "angles" in mat_p:
        mat_kwargs["angles"] = mat_p["angles"]

    has_gt = bool(rec.get("has_ground_truth", True))
    if material_model_name in ["experimental", "none", "unknown"] or not has_gt:
        print("[DATASET] Running in experimental mode without analytical ground-truth material.")
        true_mat_model = None
        psi_true_func = None
        piola_true_func = None
    else:
        try:
            true_mat_model = get_material(material_model_name, **mat_kwargs)
            psi_true_func = lambda f: true_mat_model.psi(f)
            piola_true_func = lambda f: true_mat_model.P(f)
        except Exception as e:
            print(f"[DATASET] Notice: Could not instantiate material '{material_model_name}' ({e}); setting true_mat_model = None.")
            true_mat_model = None
            psi_true_func = None
            piola_true_func = None

    # Data use in VFM
    if stress_mode == "plane_stress":
        if "F_3d" in prep_data:
            f3x3 = jnp.asarray(prep_data["F_3d"][train_load_steps_indices], dtype=jnp.float64)
        elif "lam3" in prep_data:
            lam3_train = prep_data["lam3"][train_load_steps_indices]
            f3x3_np = np.zeros((*f2x2.shape[:-2], 3, 3), dtype=np.float64)
            f3x3_np[:, :, :2, :2] = np.array(f2x2)
            f3x3_np[:, :, 2, 2] = np.array(lam3_train)
            f3x3 = jnp.asarray(f3x3_np, dtype=jnp.float64)
        elif true_mat_model is not None:
            _, solve_lambda3 = make_plane_stress_piola(true_mat_model)
            lam3_train = jax.vmap(jax.vmap(solve_lambda3))(f2x2)
            f3x3_np = np.zeros((*f2x2.shape[:-2], 3, 3), dtype=np.float64)
            f3x3_np[:, :, :2, :2] = np.array(f2x2)
            f3x3_np[:, :, 2, 2] = np.array(lam3_train)
            f3x3 = jnp.asarray(f3x3_np, dtype=jnp.float64)
        else:
            # Fallback to incompressible plane stress: lambda3 = 1 / det(F_2D)
            det_f2d = jnp.linalg.det(f2x2)
            lam3_train = 1.0 / jnp.clip(det_f2d, 1e-4, 1e4)
            f3x3_np = np.zeros((*f2x2.shape[:-2], 3, 3), dtype=np.float64)
            f3x3_np[:, :, :2, :2] = np.array(f2x2)
            f3x3_np[:, :, 2, 2] = np.array(lam3_train)
            f3x3 = jnp.asarray(f3x3_np, dtype=jnp.float64)
    else:
        f3x3 = jax.vmap(jax.vmap(fto3x3))(f2x2)

    f_neu_nodes = prep_data["f_neu"][train_load_steps_indices] 
    dNdX = prep_data["dNdX"]
    dA = prep_data["dA"]
    load_noise_std = prep_data["load_noise_std"]
    load_noise_std_steps = prep_data["load_noise_std_steps"][train_load_steps_indices] 
    loads_all = prep_data.get("reaction_forces", prep_data.get("load", None))
    loads_train = jnp.asarray(loads_all[train_load_steps_indices], dtype=jnp.float64) if loads_all is not None else None

    if args.model_mode in ["anisotropic", "aniso_unk_fiber", "aniso_unk_fiber_neg"]:
        a0_val = getattr(true_mat_model, "a0", None)
        if a0_val is None:
            a0_val = getattr(true_mat_model, "a1", None)
        if a0_val is None and "a0" in prep_data:
            cand = prep_data["a0"]
            if hasattr(cand, "dtype") and cand.dtype == object:
                cand_item = cand.item() if cand.ndim == 0 else None
                a0_val = cand_item if cand_item is not None else None
            else:
                a0_val = cand
        if a0_val is None and args.angles is not None and len(args.angles) > 0:
            deg = float(args.angles[0])
            rad = float(jnp.radians(deg)) if abs(deg) > 2.0 * float(jnp.pi) else float(deg)
            a0_val = np.array([np.cos(rad), np.sin(rad), 0.0], dtype=np.float64)

        # Validate that a0_val is a valid numeric array of size 3
        is_valid_a0 = False
        if a0_val is not None:
            try:
                a0_val_np = np.asarray(a0_val, dtype=np.float64)
                if a0_val_np.size == 3:
                    a0_val = a0_val_np
                    is_valid_a0 = True
            except Exception:
                is_valid_a0 = False

        # If not valid (e.g. isotropic model without angles), initialize random angle in [-89.9, 89.9] degrees
        if not is_valid_a0:
            angle_key = jax.random.PRNGKey(args.seed + 1000)
            rand_deg = float(jax.random.uniform(angle_key, minval=-89.9, maxval=89.9))
            rand_rad = float(jnp.radians(rand_deg))
            a0_val = np.array([np.cos(rand_rad), np.sin(rand_rad), 0.0], dtype=np.float64)
            args.angles = [rand_deg]
            config_dict["angles"] = [rand_deg]
            config_dict["a0"] = a0_val.tolist()
            with open(os.path.join(save_path, "config.json"), "w") as f:
                json.dump(config_dict, f, indent=4)
            with open(os.path.join(save_path, "config.yaml"), "w") as f:
                yaml.dump(config_dict, f, default_flow_style=False)
            print(f"🎲 No structural fiber angle provided for '{args.model_mode}'. Initialized random fiber angle: {rand_deg:.2f}° -> a0 = {a0_val.tolist()}")

        a0 = jnp.asarray(a0_val, dtype=jnp.float64)
        
        a1_cand = getattr(true_mat_model, "a1", None) if getattr(true_mat_model, "a0", None) is not None else getattr(true_mat_model, "a2", None)
        a1 = None
        if a1_cand is not None:
            try:
                a1_np = np.asarray(a1_cand, dtype=np.float64)
                if a1_np.size == 3:
                    a1 = jnp.asarray(a1_np, dtype=jnp.float64)
            except Exception:
                a1 = None

        extractor = AnisotropicFeatureExtractor(a0, a1=a1, cap_compression=args.cap_compression == 1)
        dev, vol, aniso = jax.vmap(jax.vmap(extractor.extract))(f3x3)
        I_all = jnp.concatenate([dev, vol, aniso], axis=-1)
        aniso_flat = aniso.reshape(-1, aniso.shape[-1])
    else:
        extractor = IsotropicFeatureExtractor()
        dev, vol = jax.vmap(jax.vmap(extractor.extract))(f3x3)
        I_all = jnp.concatenate([dev, vol], axis=-1)
        aniso_flat = None

    # get all data inside prep_data
    dev_flat =  dev.reshape(-1, dev.shape[-1]) 
    vol_flat = vol.reshape(-1, vol.shape[-1])
    
    aniso_z = None
    min_aniso = None
    max_aniso = None
    
    if args.resume_from:
        print(f"Resuming training from: {args.resume_from}")
        resume_dir = os.path.join(base_save_path, args.resume_from)
        I_z = jnp.load(os.path.join(resume_dir, "I_z.npy"))
        dev_z = I_z[:, :2]
        vol_z = I_z[:, 2:3] if I_z.shape[1] > 3 else I_z[:, 2:]
        if args.model_mode in ["anisotropic", "aniso_unk_fiber", "aniso_unk_fiber_neg"]:
            aniso_z = I_z[:, 3:]
            min_aniso = jnp.min(aniso_flat, axis=0)
            max_aniso = jnp.max(aniso_flat, axis=0)
    else:
        dev_z = farthest_point_sampling_with_fixed_point(dev_flat, n_ip, jnp.array([3.0, 3.0]))
        vol_z = farthest_point_sampling_with_fixed_point(vol_flat, n_ip, jnp.array([1.0]))
        I_z_list = [dev_z, vol_z]
        if args.model_mode in ["anisotropic", "aniso_unk_fiber", "aniso_unk_fiber_neg"]:
            if hasattr(extractor, "a0") and hasattr(extractor, "a1"):
                dot_val = float(jnp.dot(extractor.a0, extractor.a1))
                aniso_anchor = jnp.array([1.0, 1.0, dot_val**2], dtype=jnp.float64)
            else:
                aniso_anchor = jnp.ones(aniso_flat.shape[-1], dtype=jnp.float64)
            aniso_z = farthest_point_sampling_with_fixed_point(aniso_flat, n_ip, aniso_anchor)
            min_aniso = jnp.min(aniso_flat, axis=0)
            max_aniso = jnp.max(aniso_flat, axis=0)
            I_z_list.append(aniso_z)
        I_z = jnp.concat(I_z_list, axis = -1)
        
    plot_inducing_points(dev_z, vol_z, dev_flat, vol_flat, save_path, aniso_z=aniso_z, aniso_I=aniso_flat, feature_extractor=extractor)

    # Pre-training visualization: Invariant fields on domain (before optimization)
    try:
        print("Generating pre-training domain invariants visualization...")
        target_step = train_load_steps_indices[-1] if len(train_load_steps_indices) > 0 else -1
        plot_domain_invariants(
            prep_data=prep_data,
            save_path=save_path,
            step_idx=target_step,
            model_mode=args.model_mode,
            a0=a0_val if 'a0_val' in locals() else None,
            a1=a1 if 'a1' in locals() else None,
            make_png=True,
            save_comparison=True
        )
        print("✅ Pre-training domain invariants plot generated successfully.")
    except Exception as e:
        print(f"Warning: Could not generate pre-training domain invariants plot: {e}")

    # Pre-training visualization: Reaction forces (clean vs noisy) across load steps
    try:
        print("Generating pre-training reaction forces visualization...")
        plot_reaction_forces_noise_comparison(
            prep_data=prep_data,
            save_path=save_path,
            train_load_steps_indices=train_load_steps_indices,
            val_load_steps_indices=args.val_load_steps_indices,
            make_png=True,
            save_detailed=True
        )
        print("✅ Pre-training reaction forces plot generated successfully.")
    except Exception as e:
        print(f"Warning: Could not generate pre-training reaction forces plot: {e}")

    # Setup random key
    key = jax.random.PRNGKey(args.seed)

    hyperparameter_init = str(args.hyperparameter_init or rec.get("hyperparameter_init", "random")).lower()
    hyperparameter_amplitude_factor = float(args.hyperparameter_amplitude_factor if args.hyperparameter_amplitude_factor is not None
                                            else rec.get("hyperparameter_amplitude_factor", 10.0))
    hyperparameter_lengthscale_factor = float(args.hyperparameter_lengthscale_factor if args.hyperparameter_lengthscale_factor is not None
                                              else rec.get("hyperparameter_lengthscale_factor", 1.5))
    prior_mean = str(args.prior_mean or rec.get("prior_mean", "none")).lower()
    amplitude_prior_scale = float(args.amplitude_prior_scale if args.amplitude_prior_scale is not None
                                  else rec.get("amplitude_prior_scale", 0.0))
    freeze_amplitudes_stage1 = args.freeze_amplitudes_stage1 if args.freeze_amplitudes_stage1 is not None \
        else rec.get("freeze_amplitudes_stage1", False)
    if isinstance(freeze_amplitudes_stage1, str):
        freeze_amplitudes_stage1 = freeze_amplitudes_stage1.lower() in ["true", "1", "yes"]
    freeze_amplitudes_stage1 = bool(freeze_amplitudes_stage1)
    data_energy_scale = None
    if hyperparameter_init == "data" or prior_mean == "linear_elastic" or amplitude_prior_scale > 0:
        data_energy_scale = external_work_density(prep_data, node_type, control_mode, max(train_load_steps_indices))
        config_dict["data_energy_scale"] = data_energy_scale
    lin_prior_scale = float(args.linear_elastic_prior_scale if args.linear_elastic_prior_scale is not None
                            else rec.get("linear_elastic_prior_scale", 100.0 * (data_energy_scale or 0.1)))
    config_dict["prior_mean"] = prior_mean
    config_dict["linear_elastic_prior_scale"] = lin_prior_scale if prior_mean == "linear_elastic" else None
    if prior_mean == "linear_elastic":
        print(f"[CONFIGURATION] Linear-elastic prior mean: psi += mu (I1_bar-3)/2 + kappa (J-1)^2/2, (mu, kappa) ~ N(0, {lin_prior_scale:.3g}^2 I) a priori.")
    if hyperparameter_init == "data":
        print(f"[CONFIGURATION] Data-informed GP hyperparameters: energy density scale {data_energy_scale:.4g} "
              f"(external work / area), amplitude factor {hyperparameter_amplitude_factor}, "
              f"lengthscales = {hyperparameter_lengthscale_factor} x feature span.")
        config_dict["data_energy_scale"] = data_energy_scale
    config_dict["hyperparameter_init"] = hyperparameter_init
    config_dict["hyperparameter_amplitude_factor"] = hyperparameter_amplitude_factor
    config_dict["hyperparameter_lengthscale_factor"] = hyperparameter_lengthscale_factor
    amplitude_prior_centre = float(jnp.log(hyperparameter_amplitude_factor * data_energy_scale)) if amplitude_prior_scale > 0 else None
    if amplitude_prior_scale > 0:
        print(f"[CONFIGURATION] GP amplitude hyperprior: log sig ~ N(log {float(jnp.exp(amplitude_prior_centre)):.4g}, {amplitude_prior_scale}^2).")
    if freeze_amplitudes_stage1:
        print("[CONFIGURATION] GP amplitudes frozen during the residual stage.")
    config_dict["amplitude_prior_scale"] = amplitude_prior_scale
    config_dict["amplitude_prior_centre"] = amplitude_prior_centre
    config_dict["freeze_amplitudes_stage1"] = freeze_amplitudes_stage1
    # config.json/.yaml were written before these were resolved; write them again
    with open(os.path.join(save_path, "config.json"), "w") as f:
        json.dump(config_dict, f, indent=4)
    with open(os.path.join(save_path, "config.yaml"), "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False)

    # GP hyperparameter starts. 'random': raw values ~ N(0, 1). 'data': a broad, weakly informative prior with lengthscale
    # = b * span of the inducing features and amplitude = a * the energy density implied by the data (external work at the
    # last training step / area). A too-small or too-wiggly initial volumetric prior lets the deviatoric part absorb the
    # bulk stiffness (a wrong basin seen in the restart logs). Inducing values stay random.
    def _feature_range(component):
        z = {"dev": dev_z, "vol": vol_z, "aniso": aniso_z}[component]
        return jnp.maximum(jnp.max(z, axis=0) - jnp.min(z, axis=0), 1e-3)

    def init_hyper(kind, component, key, shape):
        if hyperparameter_init != "data":
            return jax.random.normal(key, shape)
        if kind == "sig":
            return jnp.full(shape, jnp.log(hyperparameter_amplitude_factor * data_energy_scale))
        constrained = constraint_lengthscale and "full" not in args.covariance_mode
        target = hyperparameter_lengthscale_factor * jnp.broadcast_to(_feature_range(component), shape)
        if constrained:   # l = 2 * span * sigmoid(raw) (param_version 3), so the factor must stay below 2
            frac = jnp.clip(hyperparameter_lengthscale_factor / 2.0, 1e-3, 1 - 1e-3)
            return jnp.full(shape, jnp.log(frac / (1 - frac)))
        return inv_softplus(target)

    def build_initial_params(init_key):
        """Random GP initialisation (lengthscales, amplitudes, inducing values); restarts call it with other keys."""
        # One independent key per random quantity (the former four shared keys made e.g. raw_vol_ls == raw_vol_sig)
        kn = dict(zip(["dev_ls", "dev_sig", "dev_um", "dev_uv", "vol_ls", "vol_sig", "vol_um", "vol_uv",
                       "an_ls", "an_sig", "an_um", "an_uv", "theta", "fix_x", "fix_y"], jax.random.split(init_key, 15)))
        raw_dev_z_fps = inv_softplus(dev_z - jnp.array([3.0, 3.0]))
        raw_vol_z_fps = inv_softplus(vol_z)

        raw_dev_u_mean_init = jax.random.normal(kn['dev_um'], (n_ip,)).at[0].set(0.0)
        raw_vol_u_mean_init = jax.random.normal(kn['vol_um'], (n_ip,)).at[0].set(0.0)
        
        if "full" in args.covariance_mode:
            raw_dev_u_var_init = (jax.random.normal(kn['dev_uv'], (n_ip, n_ip)) * 0.1)
            raw_dev_u_var_init = raw_dev_u_var_init.at[jnp.diag_indices(n_ip)].set(inv_softplus(args.u_var_anchor))
            raw_vol_u_var_init = (jax.random.normal(kn['vol_uv'], (n_ip, n_ip)) * 0.1)
            raw_vol_u_var_init = raw_vol_u_var_init.at[jnp.diag_indices(n_ip)].set(inv_softplus(args.u_var_anchor))
        else:
            raw_dev_u_var_init = jax.random.normal(kn['dev_uv'], (n_ip,)).at[0].set(inv_softplus(args.u_var_anchor))
            raw_vol_u_var_init = jax.random.normal(kn['vol_uv'], (n_ip,)).at[0].set(inv_softplus(args.u_var_anchor))

        aniso_kwargs = {}
        if args.model_mode in ["anisotropic", "aniso_unk_fiber", "aniso_unk_fiber_neg"]:
            raw_aniso_z_fps = inv_softplus(aniso_z)
            raw_aniso_u_mean_init = jax.random.normal(kn['an_um'], (n_ip,)).at[0].set(0.0)
            if "full" in args.covariance_mode:
                raw_aniso_u_var_init = (jax.random.normal(kn['an_uv'], (n_ip, n_ip)) * 0.1)
                raw_aniso_u_var_init = raw_aniso_u_var_init.at[jnp.diag_indices(n_ip)].set(inv_softplus(args.u_var_anchor))
            else:
                raw_aniso_u_var_init = jax.random.normal(kn['an_uv'], (n_ip,)).at[0].set(inv_softplus(args.u_var_anchor))
            aniso_dim = aniso_flat.shape[-1]
            aniso_kwargs = dict(
                raw_aniso_ls=init_hyper('ls', 'aniso', kn['an_ls'], (aniso_dim,)),
                raw_aniso_sig=init_hyper('sig', 'aniso', kn['an_sig'], ()),
                raw_aniso_z=raw_aniso_z_fps,
                raw_aniso_u_mean=raw_aniso_u_mean_init,
                raw_aniso_u_var=raw_aniso_u_var_init
            )
            if args.model_mode in ["aniso_unk_fiber", "aniso_unk_fiber_neg"]:
                if args.model_mode == "aniso_unk_fiber_neg":
                    deg = jax.random.uniform(kn['theta'], minval=-89.9, maxval=-0.1)
                else:
                    deg = jax.random.uniform(kn['theta'], minval=-89.9, maxval=89.9)
                print(f"Initializing fiber angle mean at {float(deg):.2f} degrees...")
                val = (deg / 180.0) + 0.5
                raw_theta = jnp.log(val / (1.0 - val))
                aniso_kwargs["raw_aniso_theta_mean"] = jnp.array(raw_theta)

        lin_kwargs = {}
        if prior_mean == "linear_elastic":   # start at zero moduli with a 10%-of-prior posterior std, uncorrelated
            lin_kwargs = dict(lin_mean=jnp.zeros(2), lin_chol_raw=jnp.array([inv_softplus(0.1 * lin_prior_scale), 0.0,
                                                                            inv_softplus(0.1 * lin_prior_scale)]))

        kzz_noise_kwargs = {}
        if trainable_kzz_noise:
            kzz_noise_kwargs["log_kzz_noise"] = jnp.log(jnp.array(kzz_jitter, dtype=jnp.float64))

        if is_fixed_reaction_force_noise:
            params = GPRawParams(
                # Lengthscales and signal variances (Normal(0, 1))
                raw_dev_ls=init_hyper('ls', 'dev', kn['dev_ls'], (2,)),
                raw_dev_sig=init_hyper('sig', 'dev', kn['dev_sig'], ()),
                
                # Inducing point means and variances
                raw_dev_z=raw_dev_z_fps,
                raw_dev_u_mean=raw_dev_u_mean_init,
                raw_dev_u_var=raw_dev_u_var_init,

                raw_vol_ls=init_hyper('ls', 'vol', kn['vol_ls'], (1,)),
                raw_vol_sig=init_hyper('sig', 'vol', kn['vol_sig'], ()),

                raw_vol_z=raw_vol_z_fps,        
                raw_vol_u_mean=raw_vol_u_mean_init,
                raw_vol_u_var=raw_vol_u_var_init,

                # Noise parameters (PDE residual noise)
                log_sigma_free_x=log_sigma_free_x_init,
                log_sigma_free_y=log_sigma_free_y_init,
                log_sigma_fix_x=sigma_fix_to_log_sigma_fix(load_noise_std_steps[:, 0]),
                log_sigma_fix_y=sigma_fix_to_log_sigma_fix(load_noise_std_steps[:, 1]),
                log_sigma_global=jnp.array(log_sigma0, dtype=jnp.float64),
                param_version=jnp.array(3.0),
                **lin_kwargs,
                **aniso_kwargs,
                **kzz_noise_kwargs
            )
        else :
            params = GPRawParams(
                # Lengthscales and signal variances (Normal(0, 1))
                raw_dev_ls=init_hyper('ls', 'dev', kn['dev_ls'], (2,)),
                raw_dev_sig=init_hyper('sig', 'dev', kn['dev_sig'], ()),
                
                # Inducing point means and variances
                raw_dev_z=raw_dev_z_fps,
                raw_dev_u_mean=raw_dev_u_mean_init,
                raw_dev_u_var=raw_dev_u_var_init,

                raw_vol_ls=init_hyper('ls', 'vol', kn['vol_ls'], (1,)),
                raw_vol_sig=init_hyper('sig', 'vol', kn['vol_sig'], ()),

                raw_vol_z=raw_vol_z_fps,        
                raw_vol_u_mean=raw_vol_u_mean_init,
                raw_vol_u_var=raw_vol_u_var_init,

                # Noise parameters (PDE residual noise)
                log_sigma_free_x=log_sigma_free_x_init,
                log_sigma_free_y=log_sigma_free_y_init,
                log_sigma_fix_x=jax.random.normal(kn['fix_x'], (load_noise_std_steps.shape[0],)),
                log_sigma_fix_y=jax.random.normal(kn['fix_y'], (load_noise_std_steps.shape[0],)),
                log_sigma_global=jnp.array(log_sigma0, dtype=jnp.float64),
                param_version=jnp.array(3.0),
                **lin_kwargs,
                **aniso_kwargs,
                **kzz_noise_kwargs
            )
        return params

    if args.resume_from:
        resume_dir = os.path.join(base_save_path, args.resume_from)
        best_params_dict = np.load(os.path.join(resume_dir, "best_params.npy"), allow_pickle=True).item()
        valid_keys = set(GPRawParams._fields)
        filtered_params = {k: v for k, v in best_params_dict.items() if k in valid_keys}
        params = GPRawParams(**filtered_params)
    else:
        params = build_initial_params(key)
    
    min_dev = jnp.min(dev_z, axis=0)
    min_vol = jnp.min(vol_z, axis=0)
    max_dev = jnp.max(dev_z, axis=0)
    max_vol = jnp.max(vol_z, axis=0)
    main_key = jr.PRNGKey(args.seed)

    model = SparseHyperelasticityGP(
        raw_params=params,
        I_z=I_z,
        min_dev=min_dev,
        min_vol=min_vol,
        max_dev=max_dev,
        max_vol=max_vol,
        sampling_mode=args.sampling_mode,
        beta=beta, L=args.num_rff,
        feature_extractor=extractor,
        min_aniso=min_aniso,
        max_aniso=max_aniso,
        aniso_z=aniso_z,
        covariance_mode=args.covariance_mode,
        normalize_ell=args.normalize_ell,
        u_var_anchor=args.u_var_anchor,
        kzz_jitter=kzz_jitter,
        constraint_lengthscale=constraint_lengthscale,
        lin_prior_scale=lin_prior_scale
    )




    eiv_indices = build_eiv_indices(node_type, np.asarray(cells)) if final_likelihood == "eiv" else None

    V_basis = None
    if args.vfm_mode in ["global_vf", "mix"]:
        from core.virtual_fields import build_kinematic_virtual_fields
        print(f"Building kinematically admissible virtual fields (order={args.vf_order}, mode={args.vfm_mode}, control={control_mode})...")
        V_basis = build_kinematic_virtual_fields(mesh_pos, node_type, order=args.vf_order, control_mode=control_mode)
        print(f"Constructed {V_basis.shape[0]} orthonormal virtual fields.")

    sig_fields = [f for f in ("raw_dev_sig", "raw_vol_sig", "raw_aniso_sig") if f in GPRawParams._fields]

    def amplitude_penalty(p):
        """-log of the log-normal amplitude hyperprior (up to a constant), summed over the GP components present."""
        pen = jnp.zeros(())
        for f in sig_fields:
            raw = getattr(p, f)
            if raw is not None:
                pen = pen + 0.5 * jnp.sum(((raw - amplitude_prior_centre) / amplitude_prior_scale) ** 2)
        return pen

    def make_loss_fn(loss_likelihood, loss_reaction_weight, freeze_sig=False):
        def loss_fn(p, k):
            if freeze_sig:
                p = p._replace(**{f: jax.lax.stop_gradient(getattr(p, f)) for f in sig_fields if getattr(p, f) is not None})
            loss, aux = _loss(p, k, loss_likelihood, loss_reaction_weight)
            if amplitude_prior_scale > 0:
                loss = loss + amplitude_penalty(p)
            return loss, aux
        return loss_fn

    def _loss(p, k, loss_likelihood, loss_reaction_weight):
        k_theta, k_loss = jax.random.split(k)
        if args.model_mode in ["aniso_unk_fiber", "aniso_unk_fiber_neg"]:
            theta_mean = jnp.pi * (jax.nn.sigmoid(p.raw_aniso_theta_mean) - 0.5)
            theta_sample = theta_mean
            a0 = jnp.array([jnp.cos(theta_sample), jnp.sin(theta_sample), 0.0])
            dyn_extractor = AnisotropicFeatureExtractor(a0, cap_compression=args.cap_compression == 1)
            local_model = SparseHyperelasticityGP(
                raw_params=p,
                I_z=I_z,
                min_dev=min_dev,
                min_vol=min_vol,
                max_dev=max_dev,
                max_vol=max_vol,
                sampling_mode=args.sampling_mode,
                beta=beta, L=args.num_rff,
                feature_extractor=dyn_extractor,
                min_aniso=min_aniso,
                max_aniso=max_aniso,
                aniso_z=aniso_z,
                covariance_mode=args.covariance_mode,
                normalize_ell=args.normalize_ell,
                u_var_anchor=args.u_var_anchor,
                kzz_jitter=kzz_jitter,
                constraint_lengthscale=constraint_lengthscale,
                lin_prior_scale=lin_prior_scale
            )
        else:
            local_model = model
        return total_stochastic_loss(
            p, local_model, f3x3, cells, cells.max() + 1, f_neu_nodes, node_type, dNdX, dA,
            k_loss, number_of_mci_sampling, args.normalize_ell,
            vfm_mode=args.vfm_mode, V_basis=V_basis,
            control_mode=control_mode, loads=loads_train,
            reaction_loss_weight=loss_reaction_weight,
            likelihood=loss_likelihood, eiv=eiv_indices, noise_prior_dof=noise_prior_dof, eiv_damping=eiv_damping,
            stability_weight=stability_prior_weight, stability_dirs=stability_directions
        )

    if two_stage:
        loss_fn = make_loss_fn("residual", reaction_loss_weight, freeze_sig=freeze_amplitudes_stage1)
        stage1_iterations = eiv_switch_max_iteration if eiv_switch_mode == "plateau" else eiv_switch_iteration
    else:
        loss_fn = make_loss_fn(likelihood, reaction_loss_weight)
        stage1_iterations = n_iterations

    if args.final_learning_rate is not None and args.final_learning_rate != learning_rate:
        schedule = optax.cosine_decay_schedule(
            init_value=learning_rate,
            decay_steps=stage1_iterations,
            alpha=args.final_learning_rate / learning_rate
        )
        opt = optax.adam(learning_rate=schedule)
    else:
        opt = optax.adam(learning_rate=learning_rate)
        
    opt_state = opt.init(params)

    stage2 = None
    if two_stage:
        stage2 = dict(
            loss_fn=make_loss_fn("eiv", 1.0),
            make_optimizer=lambda n_steps: optax.adam(learning_rate=optax.cosine_decay_schedule(
                init_value=eiv_learning_rate, decay_steps=max(1, n_steps), alpha=0.1)),
            mode=eiv_switch_mode, start_step=eiv_switch_iteration,
            min_step=eiv_switch_min_iteration, max_step=eiv_switch_max_iteration,
            window=plateau_window, rel_tol=plateau_rel_tol, patience=plateau_patience,
            name="EIV likelihood",
        )

    restarts = None
    if n_restarts > 1 and not args.resume_from:
        # restart 0 is the default initialisation; the others use keys folded from the seed
        restarts = dict(candidates=[params] + [build_initial_params(jax.random.fold_in(key, r)) for r in range(1, n_restarts)],
                        iterations=restart_iterations)
    
    trainer = HyperelasticGPTrainer(
        model=model,
        initial_params=params,
        loss_fn=loss_fn,
        opt_state=opt_state,
        optimizer=opt,
        save_path=save_path,
        true_mat_model=true_mat_model,
        I_z=I_z,
        I_all=I_all,
        min_dev=min_dev,
        min_vol=min_vol,
        max_dev=max_dev,
        max_vol=max_vol,
        freeze_fn=get_freeze_fn(is_fixed_reaction_force_noise, is_fixed_inducing_points, args.covariance_mode, is_free_x=is_free_x_jnp, is_free_y=is_free_y_jnp),
        seed=args.seed,
        vfm_mode=args.vfm_mode,
        free_noise_mode=free_noise_mode,
        stage2=stage2,
        restarts=restarts
    )

    meta_path = os.path.join(save_path, "metadata.json")
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r") as mf:
                mdata = json.load(mf)
            mdata["dataset_path"] = prep_dataset_path
            with open(meta_path, "w") as mf:
                json.dump(mdata, mf, indent=4)
        except Exception:
            pass

    log_info_str = f"{train_load_steps_indices}, {material_model_name}"
    import time
    start_time = time.time()
    best_params = trainer.train(n_iterations=n_iterations, main_key=main_key, log_info_str=log_info_str)
    extraction_time = time.time() - start_time

    print("Generating Training Data R2 Plot for all load steps...")
    pred_deg = float('nan')
    if args.model_mode in ["aniso_unk_fiber", "aniso_unk_fiber_neg"]:
        theta_pred = jnp.pi * (jax.nn.sigmoid(best_params.raw_aniso_theta_mean) - 0.5)
        a0_pred = jnp.array([jnp.cos(theta_pred), jnp.sin(theta_pred), 0.0])
        extractor = AnisotropicFeatureExtractor(a0_pred, cap_compression=args.cap_compression == 1)
        pred_deg = float(jnp.degrees(theta_pred))
        print(f"Predicted Fiber Angle: {pred_deg:.2f} degrees")
        import matplotlib.pyplot as plt
        plt.figure()
        plt.bar(["Predicted Angle"], [pred_deg], capsize=10)
        plt.ylabel("Angle (degrees)")
        plt.title("Learned Fiber Angle")
        plt.savefig(os.path.join(save_path, "predicted_angle.pdf"))
        plt.close()

    learned_gp = SparseHyperelasticityGP(
        raw_params=best_params, I_z=I_z, min_dev=min_dev, min_vol=min_vol, max_dev=max_dev, max_vol=max_vol,
        sampling_mode=args.sampling_mode, beta=beta, L=args.num_rff,
        feature_extractor=extractor,
        min_aniso=min_aniso,
        max_aniso=max_aniso,
        aniso_z=aniso_z,
        covariance_mode=args.covariance_mode,
        normalize_ell=args.normalize_ell,
        u_var_anchor=args.u_var_anchor,
        kzz_jitter=kzz_jitter,
        constraint_lengthscale=constraint_lengthscale,
        lin_prior_scale=lin_prior_scale
    )
    F_train_full_3x3 = load_f3x3_from_dataset(prep_data, material_model=true_mat_model)
    
    val_load_steps_indices = args.val_load_steps_indices
    test_load_steps_indices = args.test_load_steps_indices
    if val_load_steps_indices is None or test_load_steps_indices is None:
        for cand in [os.path.join(save_path, "recipe_config.yaml"), os.path.join(save_path, "config.yaml")]:
            if os.path.exists(cand):
                try:
                    with open(cand, "r") as f:
                        yd = yaml.safe_load(f)
                        if yd:
                            if val_load_steps_indices is None and "val_load_steps_indices" in yd:
                                val_load_steps_indices = yd["val_load_steps_indices"]
                            if test_load_steps_indices is None and "test_load_steps_indices" in yd:
                                test_load_steps_indices = yd["test_load_steps_indices"]
                        if val_load_steps_indices is not None and test_load_steps_indices is not None:
                            break
                except Exception:
                    pass

    r2_res = plot_training_r2(
        learned_gp, true_mat_model, F_train_full_3x3, save_path,
        train_steps=train_load_steps_indices,
        val_steps=val_load_steps_indices,
        test_steps=test_load_steps_indices
    )
    r2, rmse, coverage = r2_res[0], r2_res[1], r2_res[2]

    # Generate domain invariants plot (showing noisy observed data and smoothness)
    try:
        print("Generating Domain Invariants Plot (Noise-added observed data)...")
        target_step = train_load_steps_indices[-1] if len(train_load_steps_indices) > 0 else -1
        plot_domain_invariants(
            prep_data=prep_data,
            save_path=save_path,
            step_idx=target_step,
            model_mode=args.model_mode,
            a0=a0_pred if 'a0_pred' in locals() else (a0 if 'a0' in locals() else None),
            a1=a1 if 'a1' in locals() else None,
            make_png=True,
            save_comparison=True
        )
    except Exception as e:
        print(f"Warning: Failed to generate domain invariants plot: {e}")

    # Capture peak memory
    import resource
    import sys
    ru_maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_mb = ru_maxrss / (1024 ** 2) if sys.platform == "darwin" else ru_maxrss / 1024.0

    # Under 'eiv' the displacement noise was integrated out: store its posterior estimate at the posterior-mean
    # material in best_params (log_sigma_free_*), so saved parameters keep the displacement-noise meaning.
    if final_likelihood == "eiv":
        mean_params = learned_gp.load_params(best_params)
        mean_psi = learned_gp.psi_det
        dof_free = eiv_indices["dof_free"]
        K_ff_m, _ = jax.lax.map(lambda f_step: eiv_linearisation(mean_psi, f_step, cells, cells.max() + 1, dNdX, dA, eiv_indices), f3x3)
        r_m = jax.lax.map(lambda f_step: internal_force(mean_psi, f_step, cells, cells.max() + 1, dNdX, dA)[dof_free], f3x3)
        eps_m = jax.vmap(lambda K, r: damped_newton_step(K, r, eiv_damping))(K_ff_m, r_m)  # same step as in training
        nodal = free_noise_mode in ["nodal", "diagonal"]
        sig_dof = np.asarray(eiv_noise_estimate(eps_m, dof_free % 2, nodal_noise=nodal,
                                                sigma_global=mean_params.sigma_global, prior_dof=noise_prior_dof))
        if nodal:
            n_nodes_all = node_type.shape[0]
            g = float(mean_params.sigma_global)
            sx_n, sy_n = np.full(n_nodes_all, g), np.full(n_nodes_all, g)
            sx_n[dof_free[dof_free % 2 == 0] // 2] = sig_dof[dof_free % 2 == 0]
            sy_n[dof_free[dof_free % 2 == 1] // 2] = sig_dof[dof_free % 2 == 1]
            new_lx, new_ly = jnp.log(jnp.asarray(sx_n)), jnp.log(jnp.asarray(sy_n))
        else:
            new_lx = jnp.log(jnp.asarray(sig_dof[dof_free % 2 == 0][0]))
            new_ly = jnp.log(jnp.asarray(sig_dof[dof_free % 2 == 1][0]))
        best_params = best_params._replace(log_sigma_free_x=new_lx, log_sigma_free_y=new_ly)
        with open(os.path.join(save_path, "best_params.npy"), "wb") as f:
            jnp.save(f, best_params._asdict())

    # Capture physical parameters
    phys_params = learned_gp.load_params(best_params)
    
    primary_val_r2 = r2_res.val_metrics.get("r2")
    primary_val_rmse = r2_res.val_metrics.get("rmse")
    primary_val_ec = r2_res.val_metrics.get("ec")

    metrics = {
        "seed": args.seed,
        "extraction_time": extraction_time,
        "memory_peak_mb": peak_mb,
        "r2": r2,
        "rmse": rmse,
        "ec": coverage,
        "r2_train": r2_res.train_metrics.get("r2"),
        "rmse_train": r2_res.train_metrics.get("rmse"),
        "ec_train": r2_res.train_metrics.get("ec"),
        # Validation metrics
        "r2_val": primary_val_r2,
        "rmse_val": primary_val_rmse,
        "ec_val": primary_val_ec,
        # Synthetic energy metrics (for benchmark reference only)
        "r2_energy_val": r2_res.val_metrics.get("r2"),
        "rmse_energy_val": r2_res.val_metrics.get("rmse"),
        "ec_energy_val": r2_res.val_metrics.get("ec"),
        "r2_test": r2_res.test_metrics.get("r2"),
        "rmse_test": r2_res.test_metrics.get("rmse"),
        "ec_test": r2_res.test_metrics.get("ec"),
        "train_steps": r2_res.train_metrics.get("steps", train_load_steps_indices),
        "val_steps": r2_res.val_metrics.get("steps", val_load_steps_indices),
        "test_steps": r2_res.test_metrics.get("steps", test_load_steps_indices),
        "elbo": float(trainer.loss_components_hist["total_loss"][-1]) if trainer.loss_components_hist["total_loss"] else None,
        "ell": float(trainer.loss_components_hist["log_like"][-1]) if trainer.loss_components_hist["log_like"] else None,
        "kl": float(trainer.loss_components_hist["kl"][-1]) if trainer.loss_components_hist["kl"] else None,
        "phy": float(trainer.loss_components_hist["phy"][-1]) if trainer.loss_components_hist["phy"] else None,
        "disp_noise": float(args.disp_noise),
        "load_noise": float(args.load_noise),
        "fiber_direction": pred_deg,
        "vfm_mode": args.vfm_mode,
        "free_noise_mode": free_noise_mode,
        "sigma_free_x": float(phys_params.sigma_free_x) if np.ndim(phys_params.sigma_free_x) == 0 else np.array(phys_params.sigma_free_x).tolist(),
        "sigma_free_y": float(phys_params.sigma_free_y) if np.ndim(phys_params.sigma_free_y) == 0 else np.array(phys_params.sigma_free_y).tolist(),
        "sigma_fix_x": np.array(phys_params.sigma_fix_x).tolist(),
        "sigma_fix_y": np.array(phys_params.sigma_fix_y).tolist(),
        "sigma_global": float(phys_params.sigma_global) if phys_params.sigma_global is not None else None,
    }
    if np.ndim(phys_params.sigma_free_x) > 0:
        metrics["sigma_free_x_mean"] = float(np.mean(phys_params.sigma_free_x))
        metrics["sigma_free_y_mean"] = float(np.mean(phys_params.sigma_free_y))
    metrics["likelihood"] = likelihood
    metrics["final_likelihood"] = final_likelihood
    metrics["switch_step"] = trainer.switch_step
    metrics["switch_reason"] = trainer.switch_reason
    metrics["final_stage_convergence"] = trainer.final_monitor.report()
    if getattr(phys_params, "lin_mean", None) is not None:
        lc = np.asarray(phys_params.lin_cov); sd = np.sqrt(np.diag(lc))
        metrics["linear_elastic_mean"] = {"mu": float(phys_params.lin_mean[0]), "kappa": float(phys_params.lin_mean[1]),
                                          "mu_std": float(sd[0]), "kappa_std": float(sd[1]),
                                          "corr_mu_kappa": float(lc[0, 1] / (sd[0] * sd[1]))}
    # Material stability of the learned mean energy at the training states (reported for every run)
    stab_pen, stab_frac = ellipticity_penalty(learned_gp.psi_det, f3x3, stability_directions)
    metrics["stability_penalty"] = float(stab_pen)
    metrics["fraction_unstable"] = float(stab_frac)
    if final_likelihood == "eiv":
        # Learned noise is a displacement std; downstream validation expects a force-residual std, so also
        # report the nodal force noise it implies through the posterior-mean tangent, Cov(r) = K diag(sigma_u^2) K^T.
        n_nodes_all = node_type.shape[0]
        dof_free = eiv_indices["dof_free"]
        sx_nodes = np.broadcast_to(np.asarray(phys_params.sigma_free_x), (n_nodes_all,))
        sy_nodes = np.broadcast_to(np.asarray(phys_params.sigma_free_y), (n_nodes_all,))
        sigma_u_dof = jnp.asarray(np.where(dof_free % 2 == 0, sx_nodes[dof_free // 2], sy_nodes[dof_free // 2]))
        sigma_r_dof = np.asarray(eiv_force_equivalent_sigma(
            learned_gp.psi_det, f3x3, cells, n_nodes_all, dNdX, dA, eiv_indices, sigma_u_dof))
        metrics["sigma_u_x"] = metrics["sigma_free_x"]
        metrics["sigma_u_y"] = metrics["sigma_free_y"]
        metrics["sigma_free_x"] = float(np.sqrt(np.mean(sigma_r_dof[dof_free % 2 == 0]**2)))
        metrics["sigma_free_y"] = float(np.sqrt(np.mean(sigma_r_dof[dof_free % 2 == 1]**2)))
        print(f"[EIV] displacement noise sigma_u: x={np.mean(sx_nodes):.3e}, y={np.mean(sy_nodes):.3e} | "
              f"implied force-residual sigma_free: x={metrics['sigma_free_x']:.3e}, y={metrics['sigma_free_y']:.3e}")
    
    with open(os.path.join(save_path, "extraction_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    # 2D Spatial distribution plot of learned nodal noise
    if free_noise_mode in ["nodal", "diagonal"] or np.ndim(phys_params.sigma_free_x) > 0:
        try:
            print("Generating 2D Spatial Distribution of Learned Nodal Noise...")
            from plots.training import plot_nodal_noise_spatial_distribution
            plot_nodal_noise_spatial_distribution(
                mesh_pos=mesh_pos,
                node_type=node_type,
                sigma_free_x=phys_params.sigma_free_x,
                sigma_free_y=phys_params.sigma_free_y,
                save_path=save_path,
                control_mode=control_mode,
                cells=cells
            )
        except Exception as e:
            print(f"Warning: Failed to generate nodal noise spatial plot: {e}")

    print(f"{timestamp}_{training_config_str}")


