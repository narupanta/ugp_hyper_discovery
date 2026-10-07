import os
import sys
import resource
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
from tqdm import tqdm
from core.plotter import (
    plot_loss_analysis,
    plot_vfm_loss_analysis,
    plot_parameters_hist,
    plot_combined_validation,
    plot_energy_decomposition_validation
)
from core.model import SparseHyperelasticityGP
from core.features import AnisotropicFeatureExtractor

# Enforce mandatory 64-bit precision standard for hyperelastic computations
jax.config.update("jax_enable_x64", True)


class HyperelasticGPTrainer:
    def __init__(self, model: SparseHyperelasticityGP, initial_params, loss_fn, opt_state, optimizer, save_path, true_mat_model, I_z, I_all, min_dev, min_vol, max_dev, max_vol, freeze_fn=None, seed=None, vfm_mode: str = "linear_triangle", free_noise_mode: str = "constant", stage2: dict = None, restarts: dict = None):
        """
        stage2: optional second training stage with a different objective (e.g. residual -> EIV likelihood), dict with
            loss_fn, make_optimizer(n_steps) -> optax optimizer, name, and the switch rule:
            mode="fraction": switch at start_step;
            mode="plateau":  switch once the window-mean stage-1 loss improves by less than rel_tol*|loss| for
                             `patience` consecutive windows of `window` iterations, between min_step and max_step.
            Stage 2 starts from the best stage-1 parameters with a fresh optimizer; checkpoint selection restarts
            (the two losses are on different scales).
        restarts: optional dict(candidates=[raw params], iterations=int): before the main loop each candidate
            initialisation is trained for `iterations` with the stage-1 objective; training continues from the one
            with the lowest mean loss over its last fifth (local optima of the ELBO).
        """
        self.model = model
        self.params = initial_params
        self.opt_state = opt_state
        self.optimizer = optimizer
        self.save_path = save_path
        self.true_mat_model = true_mat_model
        self.vfm_mode = vfm_mode
        self.free_noise_mode = free_noise_mode
        
        import json
        with open(f"{self.save_path}/metadata.json", "w") as f:
            meta = {
                "covariance_mode": getattr(self.model, "covariance_mode", "diag"),
                "pos_var_mean": 1,
                "augmented_var_dist": 1,
                "normalize_ell": getattr(self.model, "normalize_ell", 0),
                "constraint_lengthscale": getattr(self.model, "constraint_lengthscale", 1),
                "vfm_mode": vfm_mode,
                "free_noise_mode": free_noise_mode
            }
            if seed is not None:
                meta["seed"] = seed
            if hasattr(self.model, "feature_extractor") and getattr(self.model.feature_extractor, "a0", None) is not None:
                meta["a0"] = np.array(self.model.feature_extractor.a0).tolist()
                if getattr(self.model.feature_extractor, "a1", None) is not None:
                    meta["a1"] = np.array(self.model.feature_extractor.a1).tolist()
            json.dump(meta, f, indent=4)

        self.I_z = I_z
        self.I_all = I_all
        self.min_dev = min_dev
        self.min_vol = min_vol
        self.max_dev = max_dev
        self.max_vol = max_vol
        self.freeze_fn = freeze_fn

        # JIT compile the single-step loss and gradient function (legacy compatibility)
        self.loss_and_grad = jax.jit(jax.value_and_grad(loss_fn, has_aux=True))
        
        # JIT compile fused block optimization loop via jax.lax.scan for GPU efficiency
        self.train_block = self._make_train_block(loss_fn, optimizer, freeze_fn)
        self.optimizer = optimizer
        self.stage2 = stage2
        self.restarts = restarts

        self.log_file_path = os.path.join(save_path, "optimization_log.txt")
        self.loss_components_hist = {
            "total_loss": [], "log_like": [], "kl": [], "phy": [], "phy2": [],
            "free_x": [], "free_y": [], "fix_x": [], "fix_y": []
        }
        self.params_hist = {
            "dev_gp_sigma_scaling": [], "vol_gp_sigma_scaling": [],
            "dev_gp_lengthscales": [], "vol_gp_lengthscales": [], 
            "dev_u_mean": [], "dev_u_var": [], "vol_u_mean": [], "vol_u_var": [], "dev_z": [], "vol_z": [],
            "aniso_gp_sigma_scaling": [], "aniso_gp_lengthscales": [],
            "aniso_u_mean": [], "aniso_u_var": [], "aniso_z": [], "aniso_theta_mean": [], "aniso_theta_var": [],
            "sigma_free_x": [], "sigma_free_y": [], "sigma_fix_x": [], "sigma_fix_y": [],
            "sigma_global": [], "kzz_noise": []
        }
        self.steps_history = []
        self.best_loss = float('inf')
        self.best_params = initial_params

    @staticmethod
    def _make_train_block(loss_fn, optimizer, freeze_fn):
        def step_fn(state, subkey):
            params_curr, opt_state_curr = state
            (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params_curr, subkey)
            if freeze_fn:
                grads = freeze_fn(grads)
            updates, opt_state_new = optimizer.update(grads, opt_state_curr)
            params_new = optax.apply_updates(params_curr, updates)
            return (params_new, opt_state_new), (loss, aux)

        @jax.jit
        def train_block(params_in, opt_state_in, keys_in):
            (params_out, opt_state_out), (losses, aux_out) = jax.lax.scan(step_fn, (params_in, opt_state_in), keys_in)
            return params_out, opt_state_out, losses, aux_out

        return train_block

    def _log(self, msg):
        print("\n" + msg)
        with open(self.log_file_path, "a") as f:
            f.write(msg + "\n")

    def _switch_to_stage2(self, step_idx, remaining, reason):
        name = self.stage2.get("name", "stage 2")
        self._log(f"=== Step {step_idx}: switching objective to {name} ({reason}; from best stage-1 parameters, "
                  f"stage-1 best loss {self.best_loss:.6f}; fresh optimizer for {remaining} iterations) ===")
        optimizer2 = self.stage2["make_optimizer"](remaining)
        self.params = self.best_params
        self.opt_state = optimizer2.init(self.params)
        self.train_block = self._make_train_block(self.stage2["loss_fn"], optimizer2, self.freeze_fn)
        self.best_loss = float('inf')
        self.stage2_active = True
        self.switch_step = step_idx

    def _run_restarts(self, main_key, block_size):
        """Short stage-1 trainings from several initialisations; keep the lowest-loss one."""
        cands, n_it = self.restarts["candidates"], int(self.restarts["iterations"])
        results = []
        for r, p in enumerate(cands):
            params, opt_state, tail = p, self.optimizer.init(p), []
            done = 0
            while done < n_it:
                b = min(block_size, n_it - done)
                keys = jr.split(main_key, b + 1)
                main_key = keys[0]
                params, opt_state, losses, _ = self.train_block(params, opt_state, keys[1:])
                done += b
                if done > 0.8 * n_it:
                    tail.append(np.asarray(losses))
            score = float(np.mean(np.concatenate(tail))) if tail else float("inf")
            if not np.isfinite(score):
                score = float("inf")
            results.append((score, r, params, opt_state))
            self._log(f"restart {r}: mean stage-1 loss over its last {int(0.2 * n_it)} of {n_it} iterations = {score:.6f}")
        score, r, params, opt_state = min(results, key=lambda x: x[0])
        self._log(f"=== continuing from restart {r} (loss {score:.6f}) ===")
        self.params, self.opt_state = params, opt_state
        self.best_params, self.best_loss = params, float("inf")
        return main_key

    def _record_metrics(self, step, loss, aux, params):
        log_like_loss, kl_loss, free_x_log_likelihood, free_y_log_likelihood, fix_x_log_likelihood, fix_y_log_likelihood, phy_loss, phys_loss2 = aux[:8]
        noise_prior = aux[8] if len(aux) > 8 else 0.0
        
        log_message = (
            f"step {step:04d} | loss={loss:.6f} | "
            f"log_like={log_like_loss:.6f} | kl={kl_loss:.6f} | free_x={free_x_log_likelihood:.6f} | "
            f"free_y={free_y_log_likelihood:.6f} | fix_x={fix_x_log_likelihood:.6f} | "
            f"fix_y={fix_y_log_likelihood:.6f} | noise_prior={float(noise_prior):.6f} | "
            f"phy={phy_loss:.6f} | phy2 ={phys_loss2:.6f}\n"
        )
        cur_params = self.model.load_params(params)
        def _clean_for_log(x):
            if hasattr(x, 'tolist'):
                x_np = np.asarray(x)
                if x_np.ndim > 0 and x_np.size > 10:
                    return f"arr(shape={x_np.shape}, mean={float(np.mean(x_np)):.4e}, min={float(np.min(x_np)):.4e}, max={float(np.max(x_np)):.4e})"
                return x.tolist()
            return x

        clean_params = jax.tree_util.tree_map(_clean_for_log, cur_params)
        log_message += f"params: {clean_params}\n"
        log_message += "-"*50 + "\n"

        self.steps_history.append(step)
        self.loss_components_hist["total_loss"].append(float(loss))
        self.loss_components_hist["log_like"].append(float(log_like_loss))
        self.loss_components_hist["kl"].append(float(kl_loss))
        self.loss_components_hist["phy"].append(float(phy_loss))
        self.loss_components_hist["phy2"].append(float(phys_loss2))
        self.loss_components_hist["free_x"].append(float(free_x_log_likelihood))
        self.loss_components_hist["free_y"].append(float(free_y_log_likelihood))
        self.loss_components_hist["fix_x"].append(float(fix_x_log_likelihood))
        self.loss_components_hist["fix_y"].append(float(fix_y_log_likelihood))
        
        self.params_hist["dev_gp_sigma_scaling"].append(cur_params.dev_sig)
        self.params_hist["vol_gp_sigma_scaling"].append(cur_params.vol_sig)
        self.params_hist["dev_gp_lengthscales"].append(cur_params.dev_ls)
        self.params_hist["vol_gp_lengthscales"].append(cur_params.vol_ls)
        self.params_hist["dev_z"].append(cur_params.dev_z)
        self.params_hist["vol_z"].append(cur_params.vol_z)
        self.params_hist["dev_u_mean"].append(cur_params.dev_u_mean)
        self.params_hist["dev_u_var"].append(cur_params.dev_u_var)
        self.params_hist["vol_u_mean"].append(cur_params.vol_u_mean)
        self.params_hist["vol_u_var"].append(cur_params.vol_u_var)
        self.params_hist["sigma_free_x"].append(cur_params.sigma_free_x)
        self.params_hist["sigma_free_y"].append(cur_params.sigma_free_y)
        self.params_hist["sigma_fix_x"].append(cur_params.sigma_fix_x)
        self.params_hist["sigma_fix_y"].append(cur_params.sigma_fix_y)
        if getattr(cur_params, "sigma_global", None) is not None:
            self.params_hist["sigma_global"].append(cur_params.sigma_global)
        if getattr(cur_params, "kzz_noise", None) is not None:
            self.params_hist["kzz_noise"].append(cur_params.kzz_noise)
        
        if hasattr(cur_params, "aniso_sig"):
            self.params_hist["aniso_gp_sigma_scaling"].append(cur_params.aniso_sig)
            self.params_hist["aniso_gp_lengthscales"].append(cur_params.aniso_ls)
            self.params_hist["aniso_u_mean"].append(cur_params.aniso_u_mean)
            self.params_hist["aniso_u_var"].append(cur_params.aniso_u_var)
            self.params_hist["aniso_z"].append(cur_params.aniso_z)

        if getattr(cur_params, "aniso_theta_mean", None) is not None:
            self.params_hist["aniso_theta_mean"].append(cur_params.aniso_theta_mean)
            if getattr(cur_params, "aniso_theta_var", None) is not None:
                self.params_hist["aniso_theta_var"].append(cur_params.aniso_theta_var)
                log_message += f"angle: {jnp.degrees(cur_params.aniso_theta_mean):.2f} ± {jnp.degrees(cur_params.aniso_theta_var):.2f}\n"
            else:
                log_message += f"angle: {jnp.degrees(cur_params.aniso_theta_mean):.2f}\n"

        with open(self.log_file_path, "a") as f:
            f.write(log_message)
        
        return {
            "loss": f"{loss:.4f}",
            "free_x": f"{free_x_log_likelihood:.4f}",
            "free_y": f"{free_y_log_likelihood:.4f}",
            "fix_x": f"{fix_x_log_likelihood:.4f}",
            "fix_y": f"{fix_y_log_likelihood:.4f}",
            "log_like": f"{log_like_loss:.4f}",
            "kl": f"{kl_loss:.4f}",
            "phy": f"{phy_loss:.4f}",
            "phy2": f"{phys_loss2:.4f}"
        }
        
    def _rebuild_model(self, raw_params) -> SparseHyperelasticityGP:
        """GP for post-training plots with exactly the training configuration (transforms, jitter, RFF count).
        In unknown-fiber mode the feature extractor uses the fiber angle learned in raw_params."""
        m = self.model
        extractor = m.feature_extractor
        raw_theta = getattr(raw_params, "raw_aniso_theta_mean", None)
        if raw_theta is not None:
            theta = jnp.pi * (jax.nn.sigmoid(raw_theta) - 0.5)
            extractor = AnisotropicFeatureExtractor(
                jnp.array([jnp.cos(theta), jnp.sin(theta), 0.0]),
                cap_compression=getattr(m.feature_extractor, "cap_compression", False))
        return SparseHyperelasticityGP(
            raw_params=raw_params, I_z=self.I_z, min_dev=self.min_dev, min_vol=self.min_vol,
            max_dev=self.max_dev, max_vol=self.max_vol, beta=m.beta,
            sampling_mode=m.sampling_mode, L=m.L,
            feature_extractor=extractor,
            min_aniso=getattr(m, 'min_aniso', None),
            max_aniso=getattr(m, 'max_aniso', None),
            aniso_z=getattr(m, 'aniso_z', None),
            covariance_mode=m.covariance_mode,
            normalize_ell=m.normalize_ell,
            u_var_anchor=m.u_var_anchor,
            kzz_jitter=m.kzz_jitter,
            constraint_lengthscale=m.constraint_lengthscale
        )

    def train(self, n_iterations, main_key, log_info_str, block_size: int = 50):
        with open(self.log_file_path, "w") as f:
            f.write(f"{log_info_str} \n Optimization Start\n" + "="*20 + "\n")

        # Decoupled checkpointing: save static dataset arrays once before optimization loop
        with open(os.path.join(self.save_path, "I_z.npy"), "wb") as f:
            jnp.save(f, self.I_z)
        with open(os.path.join(self.save_path, "I_obs_all.npy"), "wb") as f:
            jnp.save(f, self.I_all)
        with open(os.path.join(self.save_path, "best_params.npy"), "wb") as f:
            jnp.save(f, self.best_params._asdict())

        # Determine step blocks for jax.lax.scan execution
        block_size = min(max(1, block_size), max(1, n_iterations))
        if self.restarts is not None and len(self.restarts["candidates"]) > 1:
            main_key = self._run_restarts(main_key, block_size)

        pbar = tqdm(total=n_iterations, desc="Training Sparse GP (JIT Blocks)", unit="it")
        milestone_params = []

        st2 = self.stage2
        mode = st2.get("mode", "fraction") if st2 is not None else None
        self.stage2_active, self.switch_step = False, None
        window_losses, window_means, flat_windows = [], [], 0

        step_idx = 0
        while step_idx < n_iterations:
            if st2 is not None and not self.stage2_active:
                if mode == "fraction" and step_idx >= st2["start_step"]:
                    self._switch_to_stage2(step_idx, n_iterations - step_idx, "scheduled")
                elif mode == "plateau" and step_idx >= st2["max_step"]:
                    self._switch_to_stage2(step_idx, n_iterations - step_idx, "stage-1 iteration cap reached")
                elif mode == "plateau" and step_idx >= st2["min_step"] and flat_windows >= st2["patience"]:
                    self._switch_to_stage2(step_idx, n_iterations - step_idx, "stage-1 ELBO plateau")
            if st2 is not None and not self.stage2_active:
                limit = st2["start_step"] if mode == "fraction" else st2["max_step"]
                if mode == "plateau":   # end blocks on window boundaries so plateau checks are exact
                    limit = min(limit, (step_idx // st2["window"] + 1) * st2["window"])
            else:
                limit = n_iterations
            cur_block_size = min(block_size, limit - step_idx)
            keys = jr.split(main_key, cur_block_size + 1)
            main_key = keys[0]
            block_keys = keys[1:]

            # Execute entire block inside JAX XLA compiled graph without host-device sync
            self.params, self.opt_state, losses, aux_out = self.train_block(self.params, self.opt_state, block_keys)

            step_idx += cur_block_size
            pbar.update(cur_block_size)

            # Extract final metrics from the block
            loss = float(losses[-1])
            aux_step = tuple(a[-1] for a in aux_out)
            # Select checkpoints on the block-averaged loss: a single MC estimate of the negative ELBO is
            # noisy, and taking its minimum systematically favours lucky Monte Carlo draws.
            block_loss = float(jnp.mean(losses))

            # Decoupled parameter disk I/O: save only when best loss is broken at block boundary
            if block_loss < self.best_loss:
                self.best_loss = block_loss
                self.best_params = self.params
                with open(os.path.join(self.save_path, "best_params.npy"), "wb") as f:
                    jnp.save(f, self.best_params._asdict())

            # Stage-1 plateau monitor on window means of the (noisy) per-iteration loss
            if mode == "plateau" and not self.stage2_active:
                window_losses.append(np.asarray(losses))
                if step_idx % st2["window"] == 0:
                    window_means.append(float(np.mean(np.concatenate(window_losses))))
                    window_losses = []
                    if len(window_means) >= 2:
                        improvement = window_means[-2] - window_means[-1]
                        flat = improvement < st2["rel_tol"] * abs(window_means[-1])
                        flat_windows = flat_windows + 1 if flat else 0
                        with open(self.log_file_path, "a") as f:
                            f.write(f"[plateau] step {step_idx}: window mean {window_means[-1]:.4f}, improvement "
                                    f"{improvement:.4f} vs tol {st2['rel_tol'] * abs(window_means[-1]):.4f} -> "
                                    f"{'flat' if flat else 'improving'} ({flat_windows}/{st2['patience']})\n")

            # Record metrics and update progress bar (matching legacy step % 50 == 0 behavior)
            postfix = self._record_metrics(step_idx, loss, aux_step, self.params)
            pbar.set_postfix(postfix)

            # Collect milestone parameter snapshots for post-training evolution plotting (avoids blocking JIT loop)
            if step_idx % max(1, (n_iterations // 5)) == 0 and step_idx != 0 and step_idx != n_iterations:
                milestone_params.append((step_idx, self.best_params))
        pbar.close()

        # Record and print peak memory usage upon completion of the optimization loop
        self._log_memory_report()

        # Final plots and post-training evolution validation
        print("Generating training progress evolution plots...")
        for m_step, m_params in milestone_params:
            plot_model = self._rebuild_model(m_params)
            plot_combined_validation(plot_model, self.true_mat_model, self.save_path, m_step, I_obs=self.I_all)

        plot_loss_analysis(self.loss_components_hist, self.params_hist, self.steps_history, self.save_path)
        plot_parameters_hist(self.params_hist, self.steps_history, self.save_path)
        if getattr(self, "vfm_mode", "linear_triangle") in ["global_vf", "mix"]:
            plot_vfm_loss_analysis(self.loss_components_hist, self.params_hist, self.steps_history, self.save_path, self.vfm_mode)
        
        learned_gp = self._rebuild_model(self.best_params)
        plot_combined_validation(learned_gp, self.true_mat_model, self.save_path, step_idx, I_obs=self.I_all)
        
        # New Energy Validation Plots
        plot_energy_decomposition_validation(learned_gp, self.true_mat_model, self.save_path, I_obs=self.I_all)
        
        return self.best_params

    def _log_memory_report(self):
        lines = [
            "--------------------------------------------------",
            "Memory Usage Report (Extraction Training Peak):"
        ]
        # Host CPU Peak RAM usage
        try:
            ru_maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            if sys.platform == "darwin":
                peak_mb = ru_maxrss / (1024 ** 2)
            else:
                peak_mb = ru_maxrss / 1024.0
            lines.append(f"Host CPU Peak RAM: {peak_mb:.2f} MB")
        except Exception as e:
            lines.append(f"Host CPU Peak RAM: Unavailable ({e})")
            
        try:
            import psutil
            process = psutil.Process(os.getpid())
            mem_info = process.memory_info()
            lines.append(f"Host CPU Current RAM: {mem_info.rss / (1024 ** 2):.2f} MB")
        except Exception:
            pass
            
        # JAX Device Memory (GPU / TPU / CPU stats)
        try:
            devices = jax.local_devices()
            for i, dev in enumerate(devices):
                dev_str = f"Device {i} [{dev.device_kind} ({dev.platform})]:"
                if hasattr(dev, "memory_stats") and dev.memory_stats() is not None:
                    stats = dev.memory_stats()
                    cur_bytes = stats.get("bytes_in_use", 0)
                    peak_bytes = stats.get("peak_bytes_in_use", 0)
                    if peak_bytes or cur_bytes:
                        dev_str += f" Peak = {peak_bytes / (1024**2):.2f} MB | Current = {cur_bytes / (1024**2):.2f} MB"
                    else:
                        dev_str += " Memory stats reported 0 (Managed by driver/OS)"
                else:
                    dev_str += " Device memory statistics not supported"
                lines.append(dev_str)
        except Exception as e:
            lines.append(f"JAX Device Memory: Unavailable ({e})")
        lines.append("--------------------------------------------------")
        
        report_str = "\n".join(lines)
        print("\n" + report_str)
        with open(self.log_file_path, "a") as f:
            f.write(report_str + "\n")
