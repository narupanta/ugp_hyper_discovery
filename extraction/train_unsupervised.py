import os
import json
import yaml
import datetime
from pathlib import Path
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
from core.utils import fto3x3, farthest_point_sampling_with_fixed_point
from core.dataclass import GPRawParams, GPParams, GPWeights
from core.gp_component import init_component_raw_params, freeze_component_anchor, inv_softplus
from core.material_models import get_material
from core.trainer import HyperelasticGPTrainer
from core.features import IsotropicFeatureExtractor, AnisotropicFeatureExtractor
from core.datasetclass import DatasetFactory
from core.loss_function import total_stochastic_loss
from core.plotter import plot_inducing_points, plot_training_r2

def parse_args():
    parser = argparse.ArgumentParser(description="Hyperelastic Discovery GP Extraction Configuration")

    # Dataset & Model Config
    parser.add_argument('--material_model_name', type=str, default="isihara")
    parser.add_argument('--disp_noise', type=float, default=0.0001)
    parser.add_argument('--load_noise', type=float, default=0.01)
    parser.add_argument('--target_load_true_top', type=float, default=8.0)
    parser.add_argument('--asym_factor', type=float, default=0.95)
    parser.add_argument('--model_mode', type=str, default='isotropic')
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
    parser.add_argument('--kzz_jitter', type=float, default=1e-8, help="Numerical jitter added to Kzz diagonal (default 1e-8)")

    return parser.parse_args()

def sigma_fix_to_log_sigma_fix(sigma_fix):
    return jnp.log(jnp.maximum(sigma_fix, 1e-3))

def get_freeze_fn(is_fixed_noise: bool, is_fixed_z: bool, covariance_mode: str = "diag"):
    def freeze_fn(grads):
        replace_kwargs = {}
        # Zero out anchor index 0 and optionally freeze inducing coordinates
        replace_kwargs.update(freeze_component_anchor(grads, "dev", covariance_mode, is_fixed_z))
        replace_kwargs.update(freeze_component_anchor(grads, "vol", covariance_mode, is_fixed_z))

        if getattr(grads, "raw_aniso_z", None) is not None:
            is_unknown_fiber = getattr(grads, "raw_aniso_theta_mean", None) is not None
            freeze_aniso_z = is_fixed_z and (not is_unknown_fiber)
            replace_kwargs.update(freeze_component_anchor(grads, "aniso", covariance_mode, freeze_aniso_z))

        grads = grads._replace(**replace_kwargs)
        
        # Optionally freeze reaction force noise parameters
        if is_fixed_noise:
            grads = grads._replace(
                log_sigma_fix_x=jnp.zeros_like(grads.log_sigma_fix_x),
                log_sigma_fix_y=jnp.zeros_like(grads.log_sigma_fix_y)
            )
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

    # load precomputed dataset
    data_dir = "dataset/preprocessed/syn_f" if os.path.exists("dataset/preprocessed/syn_f") else "dataset/precomputed_vfm" 
    prep_dataset_path = os.path.join(data_dir, f"{material_model_name}_{disp_noise}_{load_noise}_{target_load_true_top}_{asym_factor}_{args.geometry}_{args.seed}.npz")
    if not os.path.exists(prep_dataset_path):
        fallback_path = os.path.join(data_dir, f"{material_model_name}_{disp_noise}_{load_noise}_{target_load_true_top}_{asym_factor}_{args.geometry}.npz")
        if os.path.exists(fallback_path):
            prep_dataset_path = fallback_path
        else:
            fallback_path_no_geom = os.path.join(data_dir, f"{material_model_name}_{disp_noise}_{load_noise}_{target_load_true_top}_{asym_factor}.npz")
            if os.path.exists(fallback_path_no_geom):
                prep_dataset_path = fallback_path_no_geom
    
    dataset = DatasetFactory.create("dataset/precomputed_vfm", data_path=prep_dataset_path)
    prep_data = dataset.get_data()
    f2x2 = prep_data["F"][train_load_steps_indices] 

    # Data use in VFM
    f3x3 = jax.vmap(jax.vmap(fto3x3))(f2x2)
    f_neu_nodes = prep_data["f_neu"][train_load_steps_indices] 
    node_type = np.asarray(prep_data["node_type"])
    dNdX = prep_data["dNdX"]
    dA = prep_data["dA"]
    cells = prep_data["cells"]
    load_noise_std = prep_data["load_noise_std"]
    load_noise_std_steps = prep_data["load_noise_std_steps"][train_load_steps_indices] 

    mat_kwargs = {}
    if args.angles is not None: mat_kwargs["angles"] = args.angles
    if args.dev_params is not None: mat_kwargs["dev_params"] = args.dev_params
    if args.vol_params is not None: mat_kwargs["vol_params"] = args.vol_params
    if args.aniso_params is not None: mat_kwargs["aniso_params"] = args.aniso_params
    true_mat_model = get_material(material_model_name, **mat_kwargs)
    psi_true_func = lambda f: true_mat_model.psi(f)
    piola_true_func = lambda f: true_mat_model.P(f)

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
    
    # Setup random key
    key = jax.random.PRNGKey(args.seed)
    k1, k2, k3, k4 = jax.random.split(key, 4)

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
        best_params_dict = np.load(os.path.join(resume_dir, "best_params.npy"), allow_pickle=True).item()
        valid_keys = set(GPRawParams._fields)
        filtered_params = {k: v for k, v in best_params_dict.items() if k in valid_keys}
        params = GPRawParams(**filtered_params)
    else:
        dev_raw = init_component_raw_params(
            k1, dev_flat, n_ip, anchor=jnp.array([3.0, 3.0]),
            covariance_mode=args.covariance_mode, u_var_anchor=args.u_var_anchor, is_dev=True
        )
        vol_raw = init_component_raw_params(
            k3, vol_flat, n_ip, anchor=jnp.array([1.0]),
            covariance_mode=args.covariance_mode, u_var_anchor=args.u_var_anchor, is_dev=False
        )
        dev_z, vol_z = dev_raw["z"], vol_raw["z"]
        I_z_list = [dev_z, vol_z]

        aniso_kwargs = {}
        if args.model_mode in ["anisotropic", "aniso_unk_fiber", "aniso_unk_fiber_neg"]:
            aniso_raw = init_component_raw_params(
                k4, aniso_flat, n_ip, anchor=jnp.ones(aniso_flat.shape[-1]),
                covariance_mode=args.covariance_mode, u_var_anchor=args.u_var_anchor, is_dev=False
            )
            aniso_z = aniso_raw["z"]
            min_aniso = jnp.min(aniso_flat, axis=0)
            max_aniso = jnp.max(aniso_flat, axis=0)
            I_z_list.append(aniso_z)

            aniso_kwargs = dict(
                raw_aniso_ls=aniso_raw["raw_ls"],
                raw_aniso_sig=aniso_raw["raw_sig"],
                raw_aniso_z=aniso_raw["raw_z"],
                raw_aniso_u_mean=aniso_raw["raw_u_mean"],
                raw_aniso_u_var=aniso_raw["raw_u_var"]
            )
            if args.model_mode in ["aniso_unk_fiber", "aniso_unk_fiber_neg"]:
                min_deg = -89.9
                max_deg = -0.1 if args.model_mode == "aniso_unk_fiber_neg" else 89.9
                deg = jax.random.uniform(k1, minval=min_deg, maxval=max_deg)
                print(f"Initializing fiber angle mean at {float(deg):.2f} degrees...")
                val = (deg / 180.0) + 0.5
                raw_theta = jnp.log(val / (1.0 - val))
                aniso_kwargs["raw_aniso_theta_mean"] = jnp.array(raw_theta)

        I_z = jnp.concat(I_z_list, axis=-1)

        if is_fixed_reaction_force_noise:
            log_sigma_fix_x = sigma_fix_to_log_sigma_fix(load_noise_std_steps[:, 0])
            log_sigma_fix_y = sigma_fix_to_log_sigma_fix(load_noise_std_steps[:, 1])
        else:
            log_sigma_fix_x = jax.random.normal(k3, (load_noise_std_steps.shape[0],))
            log_sigma_fix_y = jax.random.normal(k4, (load_noise_std_steps.shape[0],))

        params = GPRawParams(
            raw_dev_ls=dev_raw["raw_ls"],
            raw_dev_sig=dev_raw["raw_sig"],
            raw_dev_z=dev_raw["raw_z"],
            raw_dev_u_mean=dev_raw["raw_u_mean"],
            raw_dev_u_var=dev_raw["raw_u_var"],

            raw_vol_ls=vol_raw["raw_ls"],
            raw_vol_sig=vol_raw["raw_sig"],
            raw_vol_z=vol_raw["raw_z"],
            raw_vol_u_mean=vol_raw["raw_u_mean"],
            raw_vol_u_var=vol_raw["raw_u_var"],

            log_sigma_free_x=jnp.log(jnp.array(1.0)),
            log_sigma_free_y=jnp.log(jnp.array(1.0)),
            log_sigma_fix_x=log_sigma_fix_x,
            log_sigma_fix_y=log_sigma_fix_y,
            **aniso_kwargs
        )
        
    plot_inducing_points(dev_z, vol_z, dev_flat, vol_flat, save_path, aniso_z=aniso_z, aniso_I=aniso_flat, feature_extractor=extractor)

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
        kzz_jitter=args.kzz_jitter
    )




    def loss_fn(p, k):
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
                kzz_jitter=args.kzz_jitter
            )
        else:
            local_model = model
        return total_stochastic_loss(p, local_model, f3x3, cells, cells.max() + 1, f_neu_nodes, node_type, dNdX, dA, k_loss, number_of_mci_sampling, args.normalize_ell)

    if args.final_learning_rate is not None and args.final_learning_rate != learning_rate:
        schedule = optax.cosine_decay_schedule(
            init_value=learning_rate,
            decay_steps=n_iterations,
            alpha=args.final_learning_rate / learning_rate
        )
        opt = optax.adam(learning_rate=schedule)
    else:
        opt = optax.adam(learning_rate=learning_rate)
        
    opt_state = opt.init(params)
    
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
        freeze_fn=get_freeze_fn(is_fixed_reaction_force_noise, is_fixed_inducing_points, args.covariance_mode),
        seed=args.seed
    )

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
        kzz_jitter=args.kzz_jitter
    )
    F_train_full_3x3 = jax.vmap(jax.vmap(fto3x3))(prep_data["F"])
    
    val_load_steps_indices = args.val_load_steps_indices
    if val_load_steps_indices is None:
        for cand in [os.path.join(save_path, "recipe_config.yaml"), os.path.join(save_path, "config.yaml")]:
            if os.path.exists(cand):
                try:
                    with open(cand, "r") as f:
                        yd = yaml.safe_load(f)
                        if yd and "val_load_steps_indices" in yd:
                            val_load_steps_indices = yd["val_load_steps_indices"]
                            break
                except Exception:
                    pass

    r2_res = plot_training_r2(
        learned_gp, true_mat_model, F_train_full_3x3, save_path,
        train_steps=train_load_steps_indices,
        val_steps=val_load_steps_indices
    )
    r2, rmse, coverage = r2_res[0], r2_res[1], r2_res[2]

    # Capture peak memory
    import resource
    import sys
    ru_maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_mb = ru_maxrss / (1024 ** 2) if sys.platform == "darwin" else ru_maxrss / 1024.0

    # Capture physical parameters
    phys_params = learned_gp.load_params(best_params)
    
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
        "r2_val": r2_res.val_metrics.get("r2"),
        "rmse_val": r2_res.val_metrics.get("rmse"),
        "ec_val": r2_res.val_metrics.get("ec"),
        "train_steps": r2_res.train_metrics.get("steps", train_load_steps_indices),
        "val_steps": r2_res.val_metrics.get("steps", val_load_steps_indices),
        "elbo": float(trainer.loss_components_hist["total_loss"][-1]) if trainer.loss_components_hist["total_loss"] else None,
        "ell": float(trainer.loss_components_hist["log_like"][-1]) if trainer.loss_components_hist["log_like"] else None,
        "kl": float(trainer.loss_components_hist["kl"][-1]) if trainer.loss_components_hist["kl"] else None,
        "phy": float(trainer.loss_components_hist["phy"][-1]) if trainer.loss_components_hist["phy"] else None,
        "disp_noise": float(args.disp_noise),
        "load_noise": float(args.load_noise),
        "fiber_direction": pred_deg,
        "sigma_free_x": float(phys_params.sigma_free_x),
        "sigma_free_y": float(phys_params.sigma_free_y),
        "sigma_fix_x": np.array(phys_params.sigma_fix_x).tolist(),
        "sigma_fix_y": np.array(phys_params.sigma_fix_y).tolist(),
    }
    
    with open(os.path.join(save_path, "extraction_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"{timestamp}_{training_config_str}")


