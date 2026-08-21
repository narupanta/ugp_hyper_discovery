import jax 
import jax.numpy as jnp
from jax import config
import jax.numpy as jnp
import jax.random as jr
import matplotlib as mpl
import matplotlib.pyplot as plt
import optax
from core.model import SparseHyperelasticityGP
from core.utils import transform_input_features
from core.dataclass import GPRawParams, GPParams, GPWeights
from core.material_models import get_material
from core.trainer import HyperelasticGPTrainer
from core.features import IsotropicFeatureExtractor, AnisotropicFeatureExtractor
from core.utils import *
import datetime
import os
from tqdm import tqdm
from core.datasetclass import TractionDataset
from core.loss_function import total_stochastic_loss
from core.plotter import \
    plot_loss_analysis, \
    plot_parameters_hist, plot_inducing_points, plot_combined_validation, plot_training_r2
# helper: per-element edge-based neumann traction contribution
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import ast

def parse_args():
    parser = argparse.ArgumentParser(description="Isihara Model Dataset and Training Configuration")

    # Dataset & Model Config
    parser.add_argument('--material_model_name', type=str, default="isihara")
    parser.add_argument('--disp_noise', type=float, default=0.0001)
    parser.add_argument('--load_noise', type=float, default=0.01)
    parser.add_argument('--target_load_true_top', type=float, default=8.0)
    parser.add_argument('--asym_factor', type=float, default=0.95)
    parser.add_argument('--model_mode', type=str, default='isotropic')

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
    parser.add_argument('--n_iterations', type=int, default=1000)
    parser.add_argument('--learning_rate', type=float, default=0.01, help="Learning rate for Adam optimizer")
    parser.add_argument('--final_learning_rate', type=float, default=None, help="Final learning rate for cosine decay. If not set or equal to learning_rate, uses constant lr.")
    parser.add_argument('--geometry', type=str, default='block', help="Geometry of the specimen")
    
    # Resume training
    parser.add_argument('--resume_from', type=str, default="", help="Name of the extraction/extracted_models folder to resume from")
    
    parser.add_argument('--seed', type=int, default=42, help="Random seed for PRNGKey")
    parser.add_argument('--batch_dir', type=str, default="", help="If provided, models are saved into batch_dir/seed")
    parser.add_argument("--covariance_mode", type=str, default="diag", choices=["diag", "full", "whitened_diag", "whitened_full"], help="Covariance matrix parameterization for inducing points.")

    return parser.parse_args()

def sigma_fix_to_log_sigma_fix(sigma_fix) :
    return jnp.log(jnp.maximum(sigma_fix, 1e-3))

def inv_softplus(y):
    """Computes initial raw parameters from physical coordinates in invariant space."""
    y_safe = jnp.maximum(y, 1e-6)
    return jnp.where(y_safe > 20.0, y_safe, jnp.log(jnp.maximum(jnp.exp(y_safe) - 1.0, 1e-8)))

def get_freeze_fn(is_fixed_noise: bool, is_fixed_z: bool, covariance_mode: str = "diag"):
    def freeze_fn(grads):
        if covariance_mode == "full":
            raw_dev_u_var = grads.raw_dev_u_var.at[0, :].set(0.0).at[:, 0].set(0.0)
            raw_vol_u_var = grads.raw_vol_u_var.at[0, :].set(0.0).at[:, 0].set(0.0)
        else:
            raw_dev_u_var = grads.raw_dev_u_var.at[0].set(0.0)
            raw_vol_u_var = grads.raw_vol_u_var.at[0].set(0.0)

        # 1. ALWAYS anchor index 0 (reference free state) in BOTH training modes
        replace_kwargs = {
            "raw_dev_z": grads.raw_dev_z.at[0].set(0.0),
            "raw_vol_z": grads.raw_vol_z.at[0].set(0.0),
            "raw_dev_u_mean": grads.raw_dev_u_mean.at[0].set(0.0),
            "raw_dev_u_var": raw_dev_u_var,
            "raw_vol_u_mean": grads.raw_vol_u_mean.at[0].set(0.0),
            "raw_vol_u_var": raw_vol_u_var
        }

        if getattr(grads, "raw_aniso_z", None) is not None:
            replace_kwargs["raw_aniso_z"] = grads.raw_aniso_z.at[0].set(0.0)
            replace_kwargs["raw_aniso_u_mean"] = grads.raw_aniso_u_mean.at[0].set(0.0)
            if covariance_mode == "full":
                raw_aniso_u_var = grads.raw_aniso_u_var.at[0, :].set(0.0).at[:, 0].set(0.0)
            else:
                raw_aniso_u_var = grads.raw_aniso_u_var.at[0].set(0.0)
            replace_kwargs["raw_aniso_u_var"] = raw_aniso_u_var
            
        grads = grads._replace(**replace_kwargs)
        
        # 2. Optionally freeze reaction force noise parameters
        if is_fixed_noise:
            grads = grads._replace(
                log_sigma_fix_x=jnp.zeros_like(grads.log_sigma_fix_x),
                log_sigma_fix_y=jnp.zeros_like(grads.log_sigma_fix_y)
            )
            
        # 3. Optionally freeze ALL inducing point positions (from FPS)
        if is_fixed_z:
            replace_kwargs = {
                "raw_dev_z": jnp.zeros_like(grads.raw_dev_z),
                "raw_vol_z": jnp.zeros_like(grads.raw_vol_z)
            }
            # Anisotropic inducing points deliberately left un-frozen so they track the shifting data manifold.
            grads = grads._replace(**replace_kwargs)
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
    
    # Subfolder with datetime or batch_dir
    if args.batch_dir:
        save_path = os.path.join(args.batch_dir, str(args.seed))
    else:
        save_path = os.path.join(base_save_path, f"{timestamp}_{training_config_str}")
        
    os.makedirs(save_path, exist_ok=True)

    # load precomputed dataset
    from core.datasetclass import DatasetFactory
    data_dir = "dataset/preprocessed/syn_f" if os.path.exists("dataset/preprocessed/syn_f") else "dataset/precomputed_vfm" 
    prep_dataset_path = os.path.join(data_dir, f"{material_model_name}_{disp_noise}_{load_noise}_{target_load_true_top}_{asym_factor}_{args.geometry}.npz")
    
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

    true_mat_model = get_material(material_model_name)
    psi_true_func = lambda f: true_mat_model.psi(f)
    piola_true_func = lambda f: true_mat_model.P(f)

    if args.model_mode in ["anisotropic", "aniso_unk_fiber", "aniso_unk_fiber_neg"]:
        a0 = jnp.asarray(prep_data.get("a0", [1.0, 0.0, 0.0]))
        extractor = AnisotropicFeatureExtractor(a0, cap_compression=args.cap_compression == 1)
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
        vol_z = I_z[:, 2:]
        if args.model_mode in ["anisotropic", "aniso_unk_fiber", "aniso_unk_fiber_neg"]:
            aniso_z = I_z[:, 3:] # assuming aniso is 1D
            min_aniso = jnp.min(aniso_flat, axis=0)
            max_aniso = jnp.max(aniso_flat, axis=0)
    else:
        dev_z = farthest_point_sampling_with_fixed_point(dev_flat, n_ip, jnp.array([3.0, 3.0]))
        vol_z = farthest_point_sampling_with_fixed_point(vol_flat, n_ip, jnp.array([1.0]))
        I_z_list = [dev_z, vol_z]
        if args.model_mode in ["anisotropic", "aniso_unk_fiber", "aniso_unk_fiber_neg"]:
            aniso_z = farthest_point_sampling_with_fixed_point(aniso_flat, n_ip, jnp.array([0.0, 0.0]))
            min_aniso = jnp.min(aniso_flat, axis=0)
            max_aniso = jnp.max(aniso_flat, axis=0)
            I_z_list.append(aniso_z)
        I_z = jnp.concat(I_z_list, axis = -1)
        
    plot_inducing_points(dev_z, vol_z, dev_flat, vol_flat, save_path)

    # Setup random key
    key = jax.random.PRNGKey(args.seed)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    
    if args.resume_from:
        resume_dir = os.path.join(base_save_path, args.resume_from)
        best_params_dict = np.load(os.path.join(resume_dir, "best_params.npy"), allow_pickle=True).item()
        params = GPRawParams(**best_params_dict)
    else:
        # Initialize inducing point locations directly from FPS picking on training data
        raw_dev_z_fps = inv_softplus(dev_z - jnp.array([3.0, 3.0]))
        raw_vol_z_fps = inv_softplus(vol_z)
        
        # Anchor index 0 (free state) in initial parameters for both training modes
        raw_dev_u_mean_init = jax.random.normal(k2, (n_ip,)).at[0].set(0.0)
        raw_vol_u_mean_init = jax.random.normal(k4, (n_ip,)).at[0].set(0.0)
        
        if "full" in args.covariance_mode:
            raw_dev_u_var_init = (jax.random.normal(k2, (n_ip, n_ip)) * 0.1)
            raw_dev_u_var_init = raw_dev_u_var_init.at[jnp.diag_indices(n_ip)].set(inv_softplus(1e-8))
            raw_vol_u_var_init = (jax.random.normal(k4, (n_ip, n_ip)) * 0.1)
            raw_vol_u_var_init = raw_vol_u_var_init.at[jnp.diag_indices(n_ip)].set(inv_softplus(1e-8))
        else:
            raw_dev_u_var_init = jax.random.normal(k2, (n_ip,)).at[0].set(inv_softplus(1e-8))
            raw_vol_u_var_init = jax.random.normal(k4, (n_ip,)).at[0].set(inv_softplus(1e-8))

        aniso_kwargs = {}
        if args.model_mode in ["anisotropic", "aniso_unk_fiber", "aniso_unk_fiber_neg"]:
            raw_aniso_z_fps = inv_softplus(aniso_z)
            raw_aniso_u_mean_init = jax.random.normal(k4, (n_ip,)).at[0].set(0.0)
            if "full" in args.covariance_mode:
                raw_aniso_u_var_init = (jax.random.normal(k4, (n_ip, n_ip)) * 0.1)
                raw_aniso_u_var_init = raw_aniso_u_var_init.at[jnp.diag_indices(n_ip)].set(inv_softplus(1e-8))
            else:
                raw_aniso_u_var_init = jax.random.normal(k4, (n_ip,)).at[0].set(inv_softplus(1e-8))
            aniso_kwargs = dict(
                raw_aniso_ls=jax.random.normal(k1, (2,)),
                raw_aniso_sig=jax.random.normal(k1, ()),
                raw_aniso_z=raw_aniso_z_fps,
                raw_aniso_u_mean=raw_aniso_u_mean_init,
                raw_aniso_u_var=raw_aniso_u_var_init,
                raw_aniso_kappa=jnp.array(0.0)
            )
            if args.model_mode in ["aniso_unk_fiber", "aniso_unk_fiber_neg"]:
                deg = jax.random.uniform(k1, minval=-89.9, maxval=89.9)
                val = (deg / 180.0) + 0.5
                raw_theta = jnp.log(val / (1.0 - val))
                aniso_kwargs["raw_aniso_theta_mean"] = raw_theta

        if is_fixed_reaction_force_noise:
            params = GPRawParams(
                # Lengthscales and signal variances (Normal(0, 1))
                raw_dev_ls=jax.random.normal(k1, (2,)),
                raw_dev_sig=jax.random.normal(k1, ()),
                
                # Inducing point means and variances
                raw_dev_z=raw_dev_z_fps,
                raw_dev_u_mean=raw_dev_u_mean_init,
                raw_dev_u_var=raw_dev_u_var_init,

                raw_vol_ls=jax.random.normal(k3, (1,)),
                raw_vol_sig=jax.random.normal(k3, ()),

                raw_vol_z=raw_vol_z_fps,        
                raw_vol_u_mean=raw_vol_u_mean_init,
                raw_vol_u_var=raw_vol_u_var_init,
                raw_vol_kappa=jnp.array(0.0),



                # Noise parameters (Fixed PDE residual noise to prevent uncertainty collapse)
                log_sigma_free_x=jnp.log(jnp.array(1.0)),
                log_sigma_free_y=jnp.log(jnp.array(1.0)),
                log_sigma_fix_x=sigma_fix_to_log_sigma_fix(load_noise_std_steps[:, 0]),
                log_sigma_fix_y=sigma_fix_to_log_sigma_fix(load_noise_std_steps[:, 1]),
                **aniso_kwargs
                )
        else :
            params = GPRawParams(
                # Lengthscales and signal variances (Normal(0, 1))
                raw_dev_ls=jax.random.normal(k1, (2,)),
                raw_dev_sig=jax.random.normal(k1, ()),
                
                # Inducing point means and variances
                raw_dev_z=raw_dev_z_fps,
                raw_dev_u_mean=raw_dev_u_mean_init,
                raw_dev_u_var=raw_dev_u_var_init,

                raw_vol_ls=jax.random.normal(k3, (1,)),
                raw_vol_sig=jax.random.normal(k3, ()),

                raw_vol_z=raw_vol_z_fps,        
                raw_vol_u_mean=raw_vol_u_mean_init,
                raw_vol_u_var=raw_vol_u_var_init,
                raw_vol_kappa=jnp.array(0.0),



                # Noise parameters (Fixed PDE residual noise to prevent uncertainty collapse)
                log_sigma_free_x=jnp.log(jnp.array(1.0)),
                log_sigma_free_y=jnp.log(jnp.array(1.0)),
                log_sigma_fix_x=jax.random.normal(k3, (load_noise_std_steps.shape[0],)),
                log_sigma_fix_y=jax.random.normal(k4, (load_noise_std_steps.shape[0],)),
                **aniso_kwargs
            )
    
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
        sampling_mode="pws",
        beta=beta, L=args.num_rff,
        feature_extractor=extractor,
        min_aniso=min_aniso,
        max_aniso=max_aniso,
        aniso_z=aniso_z,
        covariance_mode=args.covariance_mode
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
                sampling_mode="pws",
                beta=beta, L=args.num_rff,
                feature_extractor=dyn_extractor,
                min_aniso=min_aniso,
                max_aniso=max_aniso,
                aniso_z=aniso_z,
                covariance_mode=args.covariance_mode
            )
        else:
            local_model = model
        return total_stochastic_loss(p, local_model, f3x3, cells, cells.max() + 1, f_neu_nodes, node_type, dNdX, dA, k_loss, number_of_mci_sampling)

    if args.model_mode in ["aniso_unk_fiber", "aniso_unk_fiber_neg"]:
        if args.model_mode == "aniso_unk_fiber_neg":
            deg = jax.random.uniform(k1, minval=-89.9, maxval=-0.1)
        else:
            deg = jax.random.uniform(k1, minval=-89.9, maxval=89.9)
            
        print(f"Initializing fiber angle mean at {float(deg):.2f} degrees...")
        val = (deg / 180.0) + 0.5
        raw_val = float(jnp.log(val / (1.0 - val)))
        
        params = params._replace(
            raw_aniso_theta_mean=jnp.array(raw_val)
        )

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
        raw_params=best_params, I_z=I_z, min_dev=min_dev, min_vol=min_vol, max_dev=max_dev, max_vol=max_vol, beta=beta, L=args.num_rff,
        feature_extractor=extractor,
        min_aniso=min_aniso,
        max_aniso=max_aniso,
        aniso_z=aniso_z,
        covariance_mode=args.covariance_mode
    )
    F_train_full_3x3 = jax.vmap(jax.vmap(fto3x3))(prep_data["F"])
    r2, rmse, coverage = plot_training_r2(learned_gp, true_mat_model, F_train_full_3x3, save_path)

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
    
    import json
    with open(os.path.join(save_path, "extraction_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"{timestamp}_{training_config_str}")


