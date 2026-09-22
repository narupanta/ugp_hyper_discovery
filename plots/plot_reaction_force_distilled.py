import os
import sys
import json
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from core.utils import deformation_gradient_element, transformation_jacobian, C_func, I1_func, I2_func, I3_func
from plots.theme import apply_style, save_figure


def eval_psi_3d(F_3d, p, a0=None, a1=None):
    """
    Evaluates hyperelastic strain energy density scalar psi(F) for isotropic and anisotropic models.
    """
    C = C_func(F_3d)
    I1 = I1_func(C)
    I2 = I2_func(C)
    I3 = I3_func(C)
    I3_safe = jnp.clip(I3, 1.0e-8, 1.0e8)

    i1_dev = I3_safe ** (-1 / 3) * I1
    i2_dev = I3_safe ** (-2 / 3) * I2
    J_val = jnp.sqrt(I3_safe)
    i1_m3 = i1_dev - 3.0
    i2_m3 = i2_dev - 3.0
    J_m1 = J_val - 1.0

    dev_p = p[:10]
    vol_p = p[10:13]

    w_dev = (
        dev_p[0] * i1_m3 + dev_p[1] * i2_m3 + dev_p[2] * i1_m3**2 +
        dev_p[3] * i1_m3 * i2_m3 + dev_p[4] * i2_m3**2 + dev_p[5] * i1_m3**3 +
        dev_p[6] * (i1_m3**2) * i2_m3 + dev_p[7] * i1_m3 * (i2_m3**2) +
        dev_p[8] * i2_m3**3 + dev_p[9] * jnp.log(jnp.maximum(i2_dev / 3.0, 1e-8))
    )
    w_vol = vol_p[0] * J_m1**2 + vol_p[1] * J_m1**4 + vol_p[2] * J_m1**6

    w_aniso = 0.0
    if a0 is not None and len(p) > 13:
        C_bar = (I3_safe ** (-1 / 3))[..., None, None] * C
        I4_bar = jnp.einsum('i,...ij,j->...', a0, C_bar, a0)
        I4_m1 = I4_bar - 1.0
        aniso_p = p[13:19]
        w_aniso = aniso_p[0] * I4_m1**2 + aniso_p[1] * I4_m1**3 + aniso_p[2] * I4_m1**4
        if a1 is not None and len(p) > 16:
            I6_bar = jnp.einsum('i,...ij,j->...', a1, C_bar, a1)
            I6_m1 = I6_bar - 1.0
            w_aniso += aniso_p[3] * I6_m1**2 + aniso_p[4] * I6_m1**3 + aniso_p[5] * I6_m1**4

    return w_dev + w_vol + w_aniso


def eval_hyperelastic_psi(F_2d, p, a0=None, a1=None):
    F_3d = jnp.eye(3, dtype=jnp.float64).at[:2, :2].set(F_2d)
    return eval_psi_3d(F_3d, p, a0, a1)


def p_3d_func(F_3d, p, a0=None, a1=None):
    return jax.grad(eval_psi_3d, argnums=0)(F_3d, p, a0, a1)


def solve_lambda3_distilled(F_2d, p, a0=None, a1=None, max_iter=8):
    det_2d = jnp.linalg.det(F_2d)
    lam3_0 = 1.0 / jnp.clip(det_2d, 1e-4, 1e4)

    def step_fn(i, lam):
        def p33_val(l):
            F_3d = jnp.array([
                [F_2d[0, 0], F_2d[0, 1], 0.0],
                [F_2d[1, 0], F_2d[1, 1], 0.0],
                [0.0,        0.0,        l]
            ])
            return p_3d_func(F_3d, p, a0, a1)[2, 2]

        p33, dp33 = jax.value_and_grad(p33_val)(lam)
        lam_next = lam - p33 / jnp.where(jnp.abs(dp33) < 1e-12, 1.0, dp33)
        return jnp.clip(lam_next, 1e-3, 100.0)

    return jax.lax.fori_loop(0, max_iter, step_fn, lam3_0)


def piola_stress_2d(F_2d, p, a0=None, a1=None, stress_mode="plane_strain"):
    if stress_mode == "plane_stress":
        lam3 = solve_lambda3_distilled(F_2d, p, a0, a1)
        F_3d = jnp.array([
            [F_2d[0, 0], F_2d[0, 1], 0.0],
            [F_2d[1, 0], F_2d[1, 1], 0.0],
            [0.0,        0.0,        lam3]
        ])
        return p_3d_func(F_3d, p, a0, a1)[:2, :2]
    else:
        F_3d = jnp.array([
            [F_2d[0, 0], F_2d[0, 1], 0.0],
            [F_2d[1, 0], F_2d[1, 1], 0.0],
            [0.0,        0.0,        1.0]
        ])
        return p_3d_func(F_3d, p, a0, a1)[:2, :2]


def get_experiment_load_noise(data_file: str, default_noise: float = 0.01) -> float:
    """Finds experiment config.yaml/json and extracts load_noise."""
    import yaml
    p = Path(data_file).resolve()
    for ancestor in [p.parent, p.parent.parent, p.parent.parent.parent, p.parent.parent.parent.parent]:
        for fname in ["config.yaml", "config.json"]:
            cfg_p = ancestor / fname
            if cfg_p.exists():
                try:
                    if fname.endswith(".yaml"):
                        with open(cfg_p, "r") as f:
                            d = yaml.safe_load(f)
                    else:
                        with open(cfg_p, "r") as f:
                            d = json.load(f)
                    if d and "load_noise" in d:
                        return float(d["load_noise"])
                except Exception:
                    pass
    return default_noise


def compute_distilled_reaction_forces(data_file: str, load_noise: float = None, batch_size: int = 64):
    """
    Computes reaction force realizations across all load steps for all distilled model parameter samples
    evaluated on the observed experimental displacement field (u_obs / u_exp / u_true).
    """
    data = np.load(data_file, allow_pickle=True)
    coords = jnp.array(data['node_coords'])
    cells = jnp.array(data['cells'])
    loads = np.array(data['loads'])   # (n_steps, 2)
    params = jnp.array(data['selected_samples']) # (n_samples, n_params)
    node_type = np.array(data['node_type'])

    if 'u_exp' in data:
        u_obs = jnp.array(data['u_exp'])
    elif 'u_true' in data:
        u_obs = jnp.array(data['u_true'])
    elif 'u_pred' in data:
        u_obs = jnp.mean(jnp.array(data['u_pred']), axis=0)
    else:
        raise KeyError(f"No displacement data found in {data_file}")

    if load_noise is None:
        load_noise = get_experiment_load_noise(data_file)

    stress_mode = str(data['stress_mode']) if 'stress_mode' in data else 'plane_strain'
    stress_mode = stress_mode.lower()

    n_samples = params.shape[0]
    n_steps, n_nodes, _ = u_obs.shape
    coords_elems = coords[cells] # (n_cells, 3, 2)
    J = transformation_jacobian(coords_elems) # (n_cells, 2, 2)
    dA = 0.5 * jnp.abs(jnp.linalg.det(J)) # (n_cells,)

    is_fix_x = jnp.array(node_type[:, 1] == 1)
    is_fix_y = jnp.array(node_type[:, 2] == 1)

    @jax.jit
    def compute_step_sample(u_step, p):
        disp_elems = u_step[cells] # (n_cells, 3, 2)
        F_cells, dNdX = deformation_gradient_element(coords_elems, disp_elems)
        P_cells = jax.vmap(lambda f: piola_stress_2d(f, p, stress_mode=stress_mode))(F_cells)
        f_elem = jnp.einsum('cij,cnj->cni', P_cells, dNdX) * dA[:, None, None]
        
        f_int = jnp.zeros((n_nodes, 2), dtype=jnp.float64)
        for a in range(3):
            f_int = f_int.at[cells[:, a]].add(f_elem[:, a])
        
        # Reaction forces balance applied loads: R = - F_int_Dirichlet
        rx = -jnp.sum(jnp.where(is_fix_x[:, None], f_int, 0.0)[:, 0])
        ry = -jnp.sum(jnp.where(is_fix_y[:, None], f_int, 0.0)[:, 1])
        return rx, ry

    compute_all_steps = jax.vmap(compute_step_sample, in_axes=(0, None))
    compute_batch = jax.vmap(lambda p: compute_all_steps(u_obs, p))

    rx_list = []
    ry_list = []
    for b_start in range(0, n_samples, batch_size):
        b_end = min(b_start + batch_size, n_samples)
        rx_b, ry_b = compute_batch(params[b_start:b_end])
        rx_list.append(np.array(rx_b))
        ry_list.append(np.array(ry_b))

    rx_all = np.concatenate(rx_list, axis=0) # (n_samples, n_steps)
    ry_all = np.concatenate(ry_list, axis=0) # (n_samples, n_steps)

    rx_pred = rx_all
    ry_pred = ry_all

    return {
        "rx_all": rx_all,
        "ry_all": ry_all,
        "rx_pred": rx_pred,
        "ry_pred": ry_pred,
        "loads": loads,
        "load_noise": load_noise,
        "n_samples": n_samples,
        "n_steps": n_steps,
        "node_type": node_type,
    }


def format_sci(val: float, precision: int = 3) -> str:
    """Formats floating point values into LaTeX scientific notation (e.g. 3.252 \times 10^{-7})."""
    if abs(val) < 1e-12:
        return "0.0"
    exp = int(np.floor(np.log10(abs(val))))
    coeff = val / (10.0**exp)
    return rf"{coeff:.{precision}f} \times 10^{{{exp}}}"


def get_experiment_test_steps(data_file: str, n_steps: int = 20) -> list:
    """Finds experiment config.yaml/json and extracts test_load_steps_indices."""
    import yaml
    p = Path(data_file).resolve()
    for ancestor in [p.parent, p.parent.parent, p.parent.parent.parent, p.parent.parent.parent.parent]:
        for fname in ["config.yaml", "config.json", "recipe_config.yaml"]:
            cfg_p = ancestor / fname
            if cfg_p.exists():
                try:
                    if fname.endswith(".yaml"):
                        with open(cfg_p, "r") as f:
                            d = yaml.safe_load(f)
                    else:
                        with open(cfg_p, "r") as f:
                            d = json.load(f)
                    if d and "test_load_steps_indices" in d and d["test_load_steps_indices"]:
                        return [int(s) for s in d["test_load_steps_indices"]]
                except Exception:
                    pass
    return [n_steps - 2, n_steps - 1]


def get_experiment_val_steps(data_file: str, n_steps: int = 20) -> list:
    """Finds experiment config.yaml/json and extracts val_load_steps_indices."""
    import yaml
    p = Path(data_file).resolve()
    for ancestor in [p.parent, p.parent.parent, p.parent.parent.parent, p.parent.parent.parent.parent]:
        for fname in ["config.yaml", "config.json", "recipe_config.yaml"]:
            cfg_p = ancestor / fname
            if cfg_p.exists():
                try:
                    if fname.endswith(".yaml"):
                        with open(cfg_p, "r") as f:
                            d = yaml.safe_load(f)
                    else:
                        with open(cfg_p, "r") as f:
                            d = json.load(f)
                    if d and "val_load_steps_indices" in d and d["val_load_steps_indices"]:
                        return [int(s) for s in d["val_load_steps_indices"] if int(s) < n_steps]
                except Exception:
                    pass
    return [s for s in [1, 3, 5, 7] if s < n_steps]


def plot_reaction_force_block(
    res: dict,
    save_path: str,
    test_steps: list = None,
    val_steps: list = None,
    alpha: float = 0.05,
    make_png: bool = True
):
    """
    Generates publication figures for Block geometry plotting pointwise reaction forces (Rx, Ry):
    1. Raw sample 95% CI: reaction_force_distilled_block.pdf/png
    2. Conformal calibrated 95% band: reaction_force_conformal_block.pdf/png
    Computes reaction force conformal scale Q_force and variance budget.
    """
    apply_style()
    from core.conformal import compute_conformal_scale, compute_variance_budget

    rx_all = res["rx_all"] # (n_samples, n_steps)
    ry_all = res["ry_all"] # (n_samples, n_steps)
    loads = res["loads"]   # (n_steps, 2)
    load_noise = res.get("load_noise", 0.01)
    n_steps = res["n_steps"]

    data_file = os.path.join(save_path, "fem_distilled_samples.npz")
    if test_steps is None:
        test_steps = get_experiment_test_steps(data_file, n_steps)
    if val_steps is None:
        val_steps = get_experiment_val_steps(data_file, n_steps)
    
    # Filter steps
    test_idx = np.array([int(s) for s in test_steps if int(s) < n_steps])
    if len(test_idx) == 0:
        test_idx = np.arange(n_steps)

    val_idx = np.array([int(s) for s in val_steps if int(s) < n_steps])
    if len(val_idx) == 0:
        val_idx = np.array([s for s in [1, 3, 5, 7] if s < n_steps])

    # Conformal calibration on validation load steps
    # Combine Rx and Ry residuals across validation steps
    r_val_true = np.stack([loads[val_idx, 0], loads[val_idx, 1]], axis=-1) # (n_val, 2)
    r_val_pred = np.stack([rx_all[:, val_idx], ry_all[:, val_idx]], axis=-1) # (n_samples, n_val, 2)

    q_force, _, std_r_param = compute_conformal_scale(r_val_true, r_val_pred, alpha=alpha)
    sigma2_param_r = float(np.mean(std_r_param**2))
    force_budget = compute_variance_budget(sigma2_param_r, float(load_noise)**2, q_force, alpha=alpha)

    # Save conformal calibration for reaction force
    calib_force_data = {
        "geometry": "block",
        "val_steps": [int(s) for s in val_idx],
        "q_force": q_force,
        "variance_budget": force_budget
    }
    with open(os.path.join(save_path, "conformal_force_calibration.json"), "w") as f:
        json.dump(calib_force_data, f, indent=4)

    # 1-indexed step labels for display
    steps_display = test_idx + 1
    n_eval_steps = len(test_idx)

    r_obs_x = loads[test_idx, 0]
    r_obs_y = loads[test_idx, 1]

    mu_rx = np.mean(rx_all[:, test_idx], axis=0)
    mu_ry = np.mean(ry_all[:, test_idx], axis=0)
    std_rx = np.std(rx_all[:, test_idx], axis=0)
    std_ry = np.std(ry_all[:, test_idx], axis=0)

    # Uncalibrated sample quantiles (95% CI)
    q_low_x, q_high_x = np.quantile(rx_all[:, test_idx], [alpha / 2.0, 1.0 - alpha / 2.0], axis=0)
    q_low_y, q_high_y = np.quantile(ry_all[:, test_idx], [alpha / 2.0, 1.0 - alpha / 2.0], axis=0)

    # Calibrated Conformal bounds
    conf_low_x = mu_rx - q_force * std_rx
    conf_high_x = mu_rx + q_force * std_rx
    conf_low_y = mu_ry - q_force * std_ry
    conf_high_y = mu_ry + q_force * std_ry

    # Compute stats helper
    def calc_stats(low_x, high_x, low_y, high_y):
        hit_x = (r_obs_x >= low_x) & (r_obs_x <= high_x)
        hit_y = (r_obs_y >= low_y) & (r_obs_y <= high_y)
        ec_rx = float(np.mean(hit_x) * 100.0)
        ec_ry = float(np.mean(hit_y) * 100.0)
        total_hits = int(np.sum(hit_x) + np.sum(hit_y))
        total_count = 2 * n_eval_steps
        total_ec = float(total_hits / total_count * 100.0)

        ss_tot_x = np.sum((r_obs_x - np.mean(r_obs_x))**2)
        ss_res_x = np.sum((r_obs_x - mu_rx)**2)
        r2_rx = float(1.0 - ss_res_x / (ss_tot_x + 1e-12)) if ss_tot_x > 1e-12 else 1.0
        rmse_rx = float(np.sqrt(np.mean((r_obs_x - mu_rx)**2)))

        ss_tot_y = np.sum((r_obs_y - np.mean(r_obs_y))**2)
        ss_res_y = np.sum((r_obs_y - mu_ry)**2)
        r2_ry = float(1.0 - ss_res_y / (ss_tot_y + 1e-12)) if ss_tot_y > 1e-12 else 1.0
        rmse_ry = float(np.sqrt(np.mean((r_obs_y - mu_ry)**2)))

        return {
            "hit_x": hit_x, "hit_y": hit_y, "ec_rx": ec_rx, "ec_ry": ec_ry,
            "total_hits": total_hits, "total_count": total_count, "total_ec": total_ec,
            "r2_rx": r2_rx, "rmse_rx": rmse_rx, "r2_ry": r2_ry, "rmse_ry": rmse_ry
        }

    raw_stats = calc_stats(q_low_x, q_high_x, q_low_y, q_high_y)
    conf_stats = calc_stats(conf_low_x, conf_high_x, conf_low_y, conf_high_y)

    dx = 0.08 if len(steps_display) <= 5 else 0.15

    # Helper to generate plot
    def render_block_plot(err_low_x, err_high_x, err_low_y, err_high_y, stats, filename, title_prefix, is_conformal=False):
        fig, ax = plt.subplots(figsize=(8.5, 6.2))
        err_x = np.vstack([mu_rx - err_low_x, err_high_x - mu_rx])
        err_y = np.vstack([mu_ry - err_low_y, err_high_y - mu_ry])

        lbl_x = r"Conformal $R_x$ ($Q_{0.95}$ Band)" if is_conformal else r"Distilled $R_x$ (Mean $\pm$ 95% CI)"
        lbl_y = r"Conformal $R_y$ ($Q_{0.95}$ Band)" if is_conformal else r"Distilled $R_y$ (Mean $\pm$ 95% CI)"

        ax.errorbar(
            steps_display - dx, mu_rx, yerr=err_x, fmt='o',
            color='#1f77b4', ecolor='#1f77b4', elinewidth=2.2, capsize=6.0, capthick=2.0,
            markersize=8.0, label=lbl_x
        )
        ax.scatter(
            steps_display - dx, r_obs_x, color='#084594', edgecolors='black',
            marker='s', s=80, zorder=5, label=r"Observed $R_{\mathrm{obs}, x}$"
        )
        ax.errorbar(
            steps_display + dx, mu_ry, yerr=err_y, fmt='o',
            color='#2ca02c', ecolor='#2ca02c', elinewidth=2.2, capsize=6.0, capthick=2.0,
            markersize=8.0, label=lbl_y
        )
        ax.scatter(
            steps_display + dx, r_obs_y, color='#006d2c', edgecolors='black',
            marker='s', s=80, zorder=5, label=r"Observed $R_{\mathrm{obs}, y}$"
        )

        ax.set_xticks(steps_display)
        ax.set_xticklabels([f"Step {s}" for s in steps_display], fontsize=12)
        ax.set_xlabel("Test Load Steps (Holdout)", fontsize=13)
        ax.set_ylabel("Reaction Force", fontsize=13)
        ax.set_title(rf"{title_prefix} Pointwise Distilled Reaction Force (Block)", fontsize=14)
        ax.grid(True, alpha=0.25, linestyle='--')
        ax.legend(loc="upper left", fontsize=10.0, framealpha=0.92, edgecolor='#cccccc')

        q_info = f" \mid Q_{{0.95}}: {q_force:.2f}" if is_conformal else ""
        metrics_text = (
            rf"$\mathbf{{{title_prefix}\ Metrics\ (95\%\ CI){q_info}:}}$" + "\n"
            rf"$R^2_{{R_x}}: {stats['r2_rx']:.4f} \mid \mathrm{{RMSE}}_{{R_x}}: {format_sci(stats['rmse_rx'])}$" + "\n"
            rf"$\mathrm{{EC}}_{{R_x}}: {int(np.sum(stats['hit_x']))}/{n_eval_steps} \ ({stats['ec_rx']:.1f}\%)$" + "\n"
            rf"$R^2_{{R_y}}: {stats['r2_ry']:.4f} \mid \mathrm{{RMSE}}_{{R_y}}: {format_sci(stats['rmse_ry'])}$" + "\n"
            rf"$\mathrm{{EC}}_{{R_y}}: {int(np.sum(stats['hit_y']))}/{n_eval_steps} \ ({stats['ec_ry']:.1f}\%)$" + "\n"
            rf"$\mathrm{{Total \ EC \ (X+Y)}}: {stats['total_hits']}/{stats['total_count']} \ ({stats['total_ec']:.1f}\%)$"
        )
        ax.text(0.52, 0.06, metrics_text, transform=ax.transAxes,
                verticalalignment='bottom', horizontalalignment='left',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.92, edgecolor='#cccccc'),
                fontsize=9.8)

        plt.tight_layout()
        os.makedirs(save_path, exist_ok=True)
        out_pdf = os.path.join(save_path, f"{filename}.pdf")
        save_figure(fig, out_pdf, make_png=make_png)
        plt.close(fig)

    # 1. Uncalibrated plot
    render_block_plot(q_low_x, q_high_x, q_low_y, q_high_y, raw_stats, "reaction_force_distilled_block", "[Uncalibrated]")
    # 2. Conformal plot
    render_block_plot(conf_low_x, conf_high_x, conf_low_y, conf_high_y, conf_stats, "reaction_force_conformal_block", "[Conformal]", is_conformal=True)

    metrics = {
        "geometry": "block",
        "test_steps": [int(s) for s in test_idx],
        "val_steps": [int(s) for s in val_idx],
        "r2_force_x": raw_stats["r2_rx"],
        "rmse_force_x": raw_stats["rmse_rx"],
        "ec_force_x": raw_stats["ec_rx"],
        "r2_force_y": raw_stats["r2_ry"],
        "rmse_force_y": raw_stats["rmse_ry"],
        "ec_force_y": raw_stats["ec_ry"],
        "total_ec": raw_stats["total_ec"],
        "conformal": {
            "q_force": q_force,
            "calibrated_ec_x": conf_stats["ec_rx"],
            "calibrated_ec_y": conf_stats["ec_ry"],
            "calibrated_total_ec": conf_stats["total_ec"],
            "variance_budget": force_budget
        },
        "n_steps": n_eval_steps,
    }
    with open(os.path.join(save_path, "reaction_force_metrics_block.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    return metrics


def plot_reaction_force_holes(
    res: dict,
    save_path: str,
    val_steps: list = None,
    alpha: float = 0.05,
    make_png: bool = True
):
    """
    Generates publication figures for Holes geometry plotting pointwise reaction force (Ry vs Load Steps 1 to 20):
    1. Raw sample 95% CI: reaction_force_distilled_holes.pdf/png
    2. Conformal calibrated 95% band using BLOCK calibrated Q_force: reaction_force_conformal_holes.pdf/png
    """
    apply_style()
    ry_all = res["ry_all"] # (n_samples, n_steps)
    loads = res["loads"]   # (n_steps, 2)
    n_steps = res["n_steps"]

    # Load block Q_force for transferability
    p_block = Path(save_path).parent / "block" / "conformal_force_calibration.json"
    q_force = 1.0
    if p_block.exists():
        try:
            with open(p_block, "r") as f:
                c_data = json.load(f)
            q_force = float(c_data.get("q_force", 1.0))
            print(f"[Conformal] Loaded Block Q_force = {q_force:.4f} for zero-shot transfer on Holes")
        except Exception:
            q_force = 1.0

    steps_display = np.arange(1, n_steps + 1)
    r_obs_y = loads[:, 1]

    mu_ry = np.mean(ry_all, axis=0)
    std_ry = np.std(ry_all, axis=0)

    # Sample quantiles (95% CI)
    q_low_y, q_high_y = np.quantile(ry_all, [alpha / 2.0, 1.0 - alpha / 2.0], axis=0)
    # Calibrated Conformal bounds using block Q_force
    conf_low_y = mu_ry - q_force * std_ry
    conf_high_y = mu_ry + q_force * std_ry

    def calc_holes_stats(low_y, high_y):
        hit_y = (r_obs_y >= low_y) & (r_obs_y <= high_y)
        ec_ry = float(np.mean(hit_y) * 100.0)
        ss_tot_y = np.sum((r_obs_y - np.mean(r_obs_y))**2)
        ss_res_y = np.sum((r_obs_y - mu_ry)**2)
        r2_ry = float(1.0 - ss_res_y / (ss_tot_y + 1e-12))
        rmse_ry = float(np.sqrt(np.mean((r_obs_y - mu_ry)**2)))
        return {"hit_y": hit_y, "ec_ry": ec_ry, "r2_ry": r2_ry, "rmse_ry": rmse_ry}

    raw_stats = calc_holes_stats(q_low_y, q_high_y)
    conf_stats = calc_holes_stats(conf_low_y, conf_high_y)

    def render_holes_plot(low_y, high_y, stats, filename, title_prefix, is_conformal=False):
        fig, ax = plt.subplots(figsize=(9.2, 6.2))
        err_y = np.vstack([mu_ry - low_y, high_y - mu_ry])
        band_lbl = rf"Conformal 95% Band ($Q={q_force:.2f}$)" if is_conformal else r"Distilled 95% Credible Band"
        pts_lbl = r"Conformal $R_y$" if is_conformal else r"Distilled $R_y$ (Mean $\pm$ 95% CI)"

        ax.plot(steps_display, mu_ry, color='#2ca02c', lw=1.5, linestyle='--', alpha=0.7)
        ax.fill_between(steps_display, low_y, high_y, color='#2ca02c', alpha=0.12, label=band_lbl)
        ax.errorbar(
            steps_display, mu_ry, yerr=err_y, fmt='o',
            color='#2ca02c', ecolor='#2ca02c', elinewidth=1.8, capsize=4.5, capthick=1.5,
            markersize=6.5, label=pts_lbl
        )
        ax.scatter(
            steps_display, r_obs_y, color='#006d2c', edgecolors='black',
            marker='s', s=45, zorder=5, label=r"Observed $R_{\mathrm{obs}, y}$"
        )

        ax.set_xticks(steps_display)
        ax.set_xlabel("Load Step (1 to 20)", fontsize=13)
        ax.set_ylabel(r"Reaction Force $R_y$ ($\mathbf{R} \cdot \mathbf{e}_1$)", fontsize=13)
        ax.set_title(rf"{title_prefix} Pointwise Distilled Reaction Force vs. Load Step (Holes)", fontsize=14)
        ax.grid(True, alpha=0.25, linestyle='--')
        ax.legend(loc="upper left", fontsize=10.2, framealpha=0.92, edgecolor='#cccccc')

        q_info = f" \mid Q_{{0.95}}: {q_force:.2f}" if is_conformal else ""
        metrics_text = (
            rf"$\mathbf{{{title_prefix}\ Metrics\ (95\%\ CI){q_info}:}}$" + "\n"
            rf"$R^2_{{R_y}}: {stats['r2_ry']:.4f}$" + "\n"
            rf"$\mathrm{{RMSE}}_{{R_y}}: {format_sci(stats['rmse_ry'])}$" + "\n"
            rf"$\mathrm{{EC}}_{{R_y}}: {int(np.sum(stats['hit_y']))}/{n_steps} \ ({stats['ec_ry']:.1f}\%)$"
        )
        ax.text(0.60, 0.06, metrics_text, transform=ax.transAxes,
                verticalalignment='bottom', horizontalalignment='left',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.92, edgecolor='#cccccc'),
                fontsize=10.2)

        plt.tight_layout()
        os.makedirs(save_path, exist_ok=True)
        out_pdf = os.path.join(save_path, f"{filename}.pdf")
        save_figure(fig, out_pdf, make_png=make_png)
        plt.close(fig)

    # 1. Uncalibrated plot
    render_holes_plot(q_low_y, q_high_y, raw_stats, "reaction_force_distilled_holes", "[Uncalibrated]")
    # 2. Conformal plot
    render_holes_plot(conf_low_y, conf_high_y, conf_stats, "reaction_force_conformal_holes", "[Conformal]", is_conformal=True)

    metrics = {
        "geometry": "holes",
        "r2_force_y": raw_stats["r2_ry"],
        "rmse_force_y": raw_stats["rmse_ry"],
        "ec_force_y": raw_stats["ec_ry"],
        "conformal": {
            "q_force": q_force,
            "calibrated_ec_y": conf_stats["ec_ry"]
        },
        "n_steps": n_steps,
    }
    with open(os.path.join(save_path, "reaction_force_metrics_holes.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    return metrics


def compute_gp_reaction_forces(data_file: str, gp_dir: str = None, step: int = 16, n_gp_samples: int = 1024, batch_size: int = 128):
    """
    Evaluates GP predicted reaction force distribution at a given load step.
    """
    from core.dataclass import GPRawParams
    from core.model import SparseHyperelasticityGP
    import jax.random as jr
    import json

    if gp_dir is None:
        p = Path(data_file).resolve()
        candidates = []
        for ancestor in [p.parent, p.parent.parent, p.parent.parent.parent, p.parent.parent.parent.parent]:
            candidates.append(ancestor / "extracted")
            candidates.append(ancestor / "extracted_model")
            dist_src = ancestor / "distilled" / "source_extraction_dir.txt"
            if dist_src.exists():
                with open(dist_src, "r") as f:
                    candidates.append(Path(f.read().strip()))
            dev_dist_src = ancestor / "distilled" / "dev_source_extraction_dir.txt"
            if dev_dist_src.exists():
                with open(dev_dist_src, "r") as f:
                    candidates.append(Path(f.read().strip()))
        for c in candidates:
            if c.exists() and (c / "best_params.npy").exists():
                gp_dir = str(c)
                break

    if gp_dir is None or not os.path.exists(os.path.join(gp_dir, "best_params.npy")):
        print(f"[WARN] GP model directory not found for {data_file}. Skipping GP comparison.")
        return None

    try:
        best_params_dict = np.load(os.path.join(gp_dir, "best_params.npy"), allow_pickle=True).item()
        valid_fields = set(GPRawParams._fields)
        gp_params = GPRawParams(**{k: v for k, v in best_params_dict.items() if k in valid_fields})
        I_z = jnp.load(os.path.join(gp_dir, "I_z.npy"))

        dev_z = I_z[:, :2]
        vol_z = I_z[:, 2:3]
        min_dev = jnp.min(dev_z, axis=0)
        min_vol = jnp.min(vol_z, axis=0)
        max_dev = jnp.max(dev_z, axis=0)
        max_vol = jnp.max(vol_z, axis=0)

        metadata_path = os.path.join(gp_dir, "metadata.json")
        cov_mode = "diag"
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                cov_mode = json.load(f).get("covariance_mode", "diag")

        gp = SparseHyperelasticityGP(
            gp_params, I_z, min_dev, min_vol, max_dev, max_vol,
            beta=1.0, covariance_mode=cov_mode
        )

        data = np.load(data_file, allow_pickle=True)
        coords = jnp.array(data["node_coords"])
        cells = jnp.array(data["cells"])
        node_type = np.array(data["node_type"])
        loads = data["loads"]
        n_steps_avail = data["u_pred"].shape[1]
        step_eval = min(step, n_steps_avail - 1)

        if "u_true" in data:
            u_step = jnp.array(data["u_true"][step_eval])
        elif "u_exp" in data:
            u_step = jnp.array(data["u_exp"][step_eval])
        else:
            u_step = jnp.array(data["u_pred"].mean(axis=0)[step_eval])

        coords_elems = coords[cells]
        J = transformation_jacobian(coords_elems)
        dA = 0.5 * jnp.abs(jnp.linalg.det(J))
        disp_elems = u_step[cells]
        F_cells, dNdX = deformation_gradient_element(coords_elems, disp_elems)
        F_3d = jnp.eye(3, dtype=jnp.float64)[None, :, :].repeat(F_cells.shape[0], axis=0).at[:, :2, :2].set(F_cells)
        stress_mode = str(data["stress_mode"]).lower() if "stress_mode" in data else "plane_strain"
        if stress_mode == "plane_stress":
            if "lam3" in data:
                F_3d = F_3d.at[:, 2, 2].set(jnp.array(data["lam3"][step_eval]))
            elif "lam3_true" in data:
                F_3d = F_3d.at[:, 2, 2].set(jnp.array(data["lam3_true"][step_eval]))

        key = jr.PRNGKey(42)
        keys = jr.split(key, n_gp_samples)

        is_fix_x = (node_type[:, 1] == 1)
        is_fix_y = (node_type[:, 2] == 1)

        def eval_one_sample(k):
            P_3d = jax.vmap(lambda f: gp.piola(f, k))(F_3d)
            P_2d = P_3d[:, :2, :2]
            f_elem = jnp.einsum('cij,cnj->cni', P_2d, dNdX) * dA[:, None, None]
            f_int = jnp.zeros((coords.shape[0], 2), dtype=jnp.float64)
            for a in range(3):
                f_int = f_int.at[cells[:, a]].add(f_elem[:, a])
            rx = -jnp.sum(jnp.where(is_fix_x[:, None], f_int, 0.0)[:, 0])
            ry = -jnp.sum(jnp.where(is_fix_y[:, None], f_int, 0.0)[:, 1])
            return rx, ry

        rx_gp_list = []
        ry_gp_list = []
        for i in range(0, n_gp_samples, batch_size):
            b_keys = keys[i:i + batch_size]
            rx_b, ry_b = jax.vmap(eval_one_sample)(b_keys)
            rx_gp_list.append(np.array(rx_b))
            ry_gp_list.append(np.array(ry_b))

        rx_gp = np.concatenate(rx_gp_list)
        ry_gp = np.concatenate(ry_gp_list)

        rx_gp_pred = rx_gp
        ry_gp_pred = ry_gp

        print(f"✅ Evaluated {n_gp_samples} GP reaction force samples at step {step_eval}")
        return {
            "rx_gp": rx_gp,
            "ry_gp": ry_gp,
            "rx_gp_pred": rx_gp_pred,
            "ry_gp_pred": ry_gp_pred,
            "step": step_eval
        }
    except Exception as e:
        print(f"[WARN] Failed computing GP reaction forces: {e}")
        return None


def plot_reaction_force_distribution_block(
    res: dict,
    save_path: str,
    res_gp: dict = None,
    step: int = 16,
    alpha: float = 0.05,
    make_png: bool = True
):
    """
    Plots the posterior distribution of predicted reaction forces (Rx and Ry) at a specified load step
    in a 2x2 grid: Top row = Distilled model, Bottom row = Gaussian Process (grey), with observed load markers.
    """
    apply_style()
    rx_all = res["rx_all"] # (n_samples, n_steps)
    ry_all = res["ry_all"] # (n_samples, n_steps)
    rx_eval = res.get("rx_pred", rx_all)
    ry_eval = res.get("ry_pred", ry_all)
    loads = res["loads"]   # (n_steps, 2)
    n_steps = res["n_steps"]
    
    if step >= n_steps:
        step = n_steps - 1

    rx_step = rx_all[:, step]
    ry_step = ry_all[:, step]
    rx_step_eval = rx_eval[:, step]
    ry_step_eval = ry_eval[:, step]
    r_obs_x = loads[step, 0]
    r_obs_y = loads[step, 1]

    mu_rx, std_rx = float(np.mean(rx_step)), float(np.std(rx_step_eval))
    q_low_x, q_high_x = np.quantile(rx_step_eval, [alpha / 2.0, 1.0 - alpha / 2.0])

    mu_ry, std_ry = float(np.mean(ry_step)), float(np.std(ry_step_eval))
    q_low_y, q_high_y = np.quantile(ry_step_eval, [alpha / 2.0, 1.0 - alpha / 2.0])

    has_gp = (res_gp is not None)
    if has_gp:
        rx_gp = res_gp["rx_gp"]
        ry_gp = res_gp["ry_gp"]
        rx_gp_eval = res_gp.get("rx_gp_pred", rx_gp)
        ry_gp_eval = res_gp.get("ry_gp_pred", ry_gp)
        mu_rx_gp, std_rx_gp = float(np.mean(rx_gp)), float(np.std(rx_gp_eval))
        q_low_x_gp, q_high_x_gp = np.quantile(rx_gp_eval, [alpha / 2.0, 1.0 - alpha / 2.0])

        mu_ry_gp, std_ry_gp = float(np.mean(ry_gp)), float(np.std(ry_gp_eval))
        q_low_y_gp, q_high_y_gp = np.quantile(ry_gp_eval, [alpha / 2.0, 1.0 - alpha / 2.0])

        fig, axes = plt.subplots(2, 2, figsize=(14.5, 9.2))

        # --- ROW 0, COL 0: Distilled Rx ---
        ax = axes[0, 0]
        ax.hist(rx_step_eval, bins=35, density=True, alpha=0.35, color='#1f77b4', edgecolor='#1f77b4', label=r"Distilled Samples ($R_x$)")
        ax.axvline(r_obs_x, color='red', lw=2.0, linestyle='-', label=rf"Observed $R_{{\mathrm{{obs}}, x}} = {r_obs_x:.4f}$")
        ax.axvline(mu_rx, color='#1f77b4', lw=1.8, linestyle='--', label=rf"Distilled Mean: {mu_rx:.4f}")
        ax.axvline(q_low_x, color='#4ba3e3', lw=1.4, linestyle=':', label=rf"Distilled $q_{{0.025}}, q_{{0.975}}$")
        ax.axvline(q_high_x, color='#4ba3e3', lw=1.4, linestyle=':')
        ax.axvspan(q_low_x, q_high_x, color='#1f77b4', alpha=0.12, label=r"Distilled 95% CI")

        stats_text = (
            rf"$\mathbf{{Distilled \ R_x \ (Step \ {step})}}$" + "\n"
            rf"$\mathrm{{Mean}}: {mu_rx:.4f}$" + "\n"
            rf"$\mathrm{{Std}}: {format_sci(std_rx)}$" + "\n"
            rf"$95\%\ \mathrm{{CI}}: [{q_low_x:.4f}, \ {q_high_x:.4f}]$" + "\n"
            rf"$R_{{\mathrm{{obs}}, x}}: {r_obs_x:.4f}$"
        )
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.9, edgecolor='#cccccc'),
                fontsize=9.2)
        ax.set_xlabel(r"Reaction Force $R_x$", fontsize=13)
        ax.set_ylabel("Density", fontsize=13)
        ax.set_title(rf"Distilled Model: Reaction Force $R_x$ (Step {step}, Block)", fontsize=13)
        ax.grid(True, alpha=0.25, linestyle='--')
        ax.legend(loc="upper right", fontsize=8.8, framealpha=0.92, edgecolor='#cccccc')

        # --- ROW 0, COL 1: Distilled Ry ---
        ax = axes[0, 1]
        ax.hist(ry_step_eval, bins=35, density=True, alpha=0.35, color='#2ca02c', edgecolor='#2ca02c', label=r"Distilled Samples ($R_y$)")
        ax.axvline(r_obs_y, color='red', lw=2.0, linestyle='-', label=rf"Observed $R_{{\mathrm{{obs}}, y}} = {r_obs_y:.4f}$")
        ax.axvline(mu_ry, color='#2ca02c', lw=1.8, linestyle='--', label=rf"Distilled Mean: {mu_ry:.4f}")
        ax.axvline(q_low_y, color='#74c476', lw=1.4, linestyle=':', label=rf"Distilled $q_{{0.025}}, q_{{0.975}}$")
        ax.axvline(q_high_y, color='#74c476', lw=1.4, linestyle=':')
        ax.axvspan(q_low_y, q_high_y, color='#2ca02c', alpha=0.12, label=r"Distilled 95% CI")

        stats_text = (
            rf"$\mathbf{{Distilled \ R_y \ (Step \ {step})}}$" + "\n"
            rf"$\mathrm{{Mean}}: {mu_ry:.4f}$" + "\n"
            rf"$\mathrm{{Std}}: {format_sci(std_ry)}$" + "\n"
            rf"$95\%\ \mathrm{{CI}}: [{q_low_y:.4f}, \ {q_high_y:.4f}]$" + "\n"
            rf"$R_{{\mathrm{{obs}}, y}}: {r_obs_y:.4f}$"
        )
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.9, edgecolor='#cccccc'),
                fontsize=9.2)
        ax.set_xlabel(r"Reaction Force $R_y$", fontsize=13)
        ax.set_ylabel("Density", fontsize=13)
        ax.set_title(rf"Distilled Model: Reaction Force $R_y$ (Step {step}, Block)", fontsize=13)
        ax.grid(True, alpha=0.25, linestyle='--')
        ax.legend(loc="upper right", fontsize=8.8, framealpha=0.92, edgecolor='#cccccc')

        # --- ROW 1, COL 0: GP Rx (Grey) ---
        ax = axes[1, 0]
        ax.hist(rx_gp_eval, bins=35, density=True, alpha=0.35, color='#7f7f7f', edgecolor='#555555', label=r"GP Samples ($R_x$)")
        ax.axvline(r_obs_x, color='red', lw=2.0, linestyle='-', label=rf"Observed $R_{{\mathrm{{obs}}, x}} = {r_obs_x:.4f}$")
        ax.axvline(mu_rx_gp, color='#444444', lw=1.8, linestyle='--', label=rf"GP Mean: {mu_rx_gp:.4f}")
        ax.axvline(q_low_x_gp, color='#888888', lw=1.4, linestyle=':', label=rf"GP $q_{{0.025}}, q_{{0.975}}$")
        ax.axvline(q_high_x_gp, color='#888888', lw=1.4, linestyle=':')
        ax.axvspan(q_low_x_gp, q_high_x_gp, color='#7f7f7f', alpha=0.12, label=r"GP 95% CI")

        stats_text = (
            rf"$\mathbf{{Gaussian \ Process \ R_x \ (Step \ {step})}}$" + "\n"
            rf"$\mathrm{{Mean}}: {mu_rx_gp:.4f}$" + "\n"
            rf"$\mathrm{{Std}}: {format_sci(std_rx_gp)}$" + "\n"
            rf"$95\%\ \mathrm{{CI}}: [{q_low_x_gp:.4f}, \ {q_high_x_gp:.4f}]$" + "\n"
            rf"$R_{{\mathrm{{obs}}, x}}: {r_obs_x:.4f}$"
        )
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.9, edgecolor='#cccccc'),
                fontsize=9.2)
        ax.set_xlabel(r"Reaction Force $R_x$", fontsize=13)
        ax.set_ylabel("Density", fontsize=13)
        ax.set_title(rf"Gaussian Process: Reaction Force $R_x$ (Step {step}, Block)", fontsize=13)
        ax.grid(True, alpha=0.25, linestyle='--')
        ax.legend(loc="upper right", fontsize=8.8, framealpha=0.92, edgecolor='#cccccc')

        # --- ROW 1, COL 1: GP Ry (Grey) ---
        ax = axes[1, 1]
        ax.hist(ry_gp_eval, bins=35, density=True, alpha=0.35, color='#7f7f7f', edgecolor='#555555', label=r"GP Samples ($R_y$)")
        ax.axvline(r_obs_y, color='red', lw=2.0, linestyle='-', label=rf"Observed $R_{{\mathrm{{obs}}, y}} = {r_obs_y:.4f}$")
        ax.axvline(mu_ry_gp, color='#444444', lw=1.8, linestyle='--', label=rf"GP Mean: {mu_ry_gp:.4f}")
        ax.axvline(q_low_y_gp, color='#888888', lw=1.4, linestyle=':', label=rf"GP $q_{{0.025}}, q_{{0.975}}$")
        ax.axvline(q_high_y_gp, color='#888888', lw=1.4, linestyle=':')
        ax.axvspan(q_low_y_gp, q_high_y_gp, color='#7f7f7f', alpha=0.12, label=r"GP 95% CI")

        stats_text = (
            rf"$\mathbf{{Gaussian \ Process \ R_y \ (Step \ {step})}}$" + "\n"
            rf"$\mathrm{{Mean}}: {mu_ry_gp:.4f}$" + "\n"
            rf"$\mathrm{{Std}}: {format_sci(std_ry_gp)}$" + "\n"
            rf"$95\%\ \mathrm{{CI}}: [{q_low_y_gp:.4f}, \ {q_high_y_gp:.4f}]$" + "\n"
            rf"$R_{{\mathrm{{obs}}, y}}: {r_obs_y:.4f}$"
        )
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.9, edgecolor='#cccccc'),
                fontsize=9.2)
        ax.set_xlabel(r"Reaction Force $R_y$", fontsize=13)
        ax.set_ylabel("Density", fontsize=13)
        ax.set_title(rf"Gaussian Process: Reaction Force $R_y$ (Step {step}, Block)", fontsize=13)
        ax.grid(True, alpha=0.25, linestyle='--')
        ax.legend(loc="upper right", fontsize=8.8, framealpha=0.92, edgecolor='#cccccc')

    else:
        fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.5))
        # Panel 1: Rx
        ax = axes[0]
        ax.hist(rx_step_eval, bins=35, density=True, alpha=0.35, color='#1f77b4', edgecolor='#1f77b4', label=r"Distilled Samples ($R_x$)")
        ax.axvline(r_obs_x, color='red', lw=2.0, linestyle='-', label=rf"Observed $R_{{\mathrm{{obs}}, x}} = {r_obs_x:.4f}$")
        ax.axvline(mu_rx, color='#1f77b4', lw=1.8, linestyle='--', label=rf"Distilled Mean: {mu_rx:.4f}")
        ax.axvline(q_low_x, color='#4ba3e3', lw=1.4, linestyle=':', label=rf"Distilled $q_{{0.025}}, q_{{0.975}}$")
        ax.axvline(q_high_x, color='#4ba3e3', lw=1.4, linestyle=':')
        ax.axvspan(q_low_x, q_high_x, color='#1f77b4', alpha=0.12, label=r"Distilled 95% CI")
        stats_text = (
            rf"$\mathbf{{Distilled \ R_x \ (Step \ {step})}}$" + "\n"
            rf"$\mathrm{{Mean}}: {mu_rx:.4f}$" + "\n"
            rf"$\mathrm{{Std}}: {format_sci(std_rx)}$" + "\n"
            rf"$95\%\ \mathrm{{CI}}: [{q_low_x:.4f}, \ {q_high_x:.4f}]$" + "\n"
            rf"$R_{{\mathrm{{obs}}, x}}: {r_obs_x:.4f}$"
        )
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.9, edgecolor='#cccccc'), fontsize=9.2)
        ax.set_xlabel(r"Reaction Force $R_x$", fontsize=13)
        ax.set_ylabel("Density", fontsize=13)
        ax.set_title(rf"Reaction Force $R_x$ Distribution (Step {step}, Block)", fontsize=13)
        ax.grid(True, alpha=0.25, linestyle='--')
        ax.legend(loc="upper right", fontsize=8.8, framealpha=0.92, edgecolor='#cccccc')

        # Panel 2: Ry
        ax = axes[1]
        ax.hist(ry_step_eval, bins=35, density=True, alpha=0.35, color='#2ca02c', edgecolor='#2ca02c', label=r"Distilled Samples ($R_y$)")
        ax.axvline(r_obs_y, color='red', lw=2.0, linestyle='-', label=rf"Observed $R_{{\mathrm{{obs}}, y}} = {r_obs_y:.4f}$")
        ax.axvline(mu_ry, color='#2ca02c', lw=1.8, linestyle='--', label=rf"Distilled Mean: {mu_ry:.4f}")
        ax.axvline(q_low_y, color='#74c476', lw=1.4, linestyle=':', label=rf"Distilled $q_{{0.025}}, q_{{0.975}}$")
        ax.axvline(q_high_y, color='#74c476', lw=1.4, linestyle=':')
        ax.axvspan(q_low_y, q_high_y, color='#2ca02c', alpha=0.12, label=r"Distilled 95% CI")
        stats_text = (
            rf"$\mathbf{{Distilled \ R_y \ (Step \ {step})}}$" + "\n"
            rf"$\mathrm{{Mean}}: {mu_ry:.4f}$" + "\n"
            rf"$\mathrm{{Std}}: {format_sci(std_ry)}$" + "\n"
            rf"$95\%\ \mathrm{{CI}}: [{q_low_y:.4f}, \ {q_high_y:.4f}]$" + "\n"
            rf"$R_{{\mathrm{{obs}}, y}}: {r_obs_y:.4f}$"
        )
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.9, edgecolor='#cccccc'), fontsize=9.2)
        ax.set_xlabel(r"Reaction Force $R_y$", fontsize=13)
        ax.set_ylabel("Density", fontsize=13)
        ax.set_title(rf"Reaction Force $R_y$ Distribution (Step {step}, Block)", fontsize=13)
        ax.grid(True, alpha=0.25, linestyle='--')
        ax.legend(loc="upper right", fontsize=8.8, framealpha=0.92, edgecolor='#cccccc')

    plt.tight_layout()
    os.makedirs(save_path, exist_ok=True)
    out_pdf = os.path.join(save_path, "reaction_force_distribution_block.pdf")
    save_figure(fig, out_pdf, make_png=make_png)
    plt.close(fig)


def plot_reaction_force_distribution_holes(
    res: dict,
    save_path: str,
    res_gp: dict = None,
    step: int = 16,
    alpha: float = 0.05,
    make_png: bool = True
):
    """
    Plots the posterior distribution of predicted reaction force Ry at a specified load step
    in a 2x1 grid: Top = Distilled model (green), Bottom = Gaussian Process (grey), with observed load marker.
    """
    apply_style()
    ry_all = res["ry_all"] # (n_samples, n_steps)
    ry_eval = res.get("ry_pred", ry_all)
    loads = res["loads"]   # (n_steps, 2)
    n_steps = res["n_steps"]
    
    if step >= n_steps:
        step = n_steps - 1

    ry_step = ry_all[:, step]
    ry_step_eval = ry_eval[:, step]
    r_obs_y = loads[step, 1]

    mu_ry, std_ry = float(np.mean(ry_step)), float(np.std(ry_step_eval))
    q_low_y, q_high_y = np.quantile(ry_step_eval, [alpha / 2.0, 1.0 - alpha / 2.0])

    has_gp = (res_gp is not None)
    if has_gp:
        ry_gp = res_gp["ry_gp"]
        ry_gp_eval = res_gp.get("ry_gp_pred", ry_gp)
        mu_ry_gp, std_ry_gp = float(np.mean(ry_gp)), float(np.std(ry_gp_eval))
        q_low_y_gp, q_high_y_gp = np.quantile(ry_gp_eval, [alpha / 2.0, 1.0 - alpha / 2.0])

        fig, axes = plt.subplots(2, 1, figsize=(7.8, 9.2))

        # --- ROW 0: Distilled Ry ---
        ax = axes[0]
        ax.hist(ry_step_eval, bins=35, density=True, alpha=0.35, color='#2ca02c', edgecolor='#2ca02c', label=r"Distilled Samples ($R_y$)")
        ax.axvline(r_obs_y, color='red', lw=2.0, linestyle='-', label=rf"Observed $R_{{\mathrm{{obs}}, y}} = {r_obs_y:.4f}$")
        ax.axvline(mu_ry, color='#2ca02c', lw=1.8, linestyle='--', label=rf"Distilled Mean: {mu_ry:.4f}")
        ax.axvline(q_low_y, color='#74c476', lw=1.4, linestyle=':', label=rf"Distilled $q_{{0.025}}, q_{{0.975}}$")
        ax.axvline(q_high_y, color='#74c476', lw=1.4, linestyle=':')
        ax.axvspan(q_low_y, q_high_y, color='#2ca02c', alpha=0.12, label=r"Distilled 95% CI")

        stats_text = (
            rf"$\mathbf{{Distilled \ R_y \ (Step \ {step})}}$" + "\n"
            rf"$\mathrm{{Mean}}: {mu_ry:.4f}$" + "\n"
            rf"$\mathrm{{Std}}: {format_sci(std_ry)}$" + "\n"
            rf"$95\%\ \mathrm{{CI}}: [{q_low_y:.4f}, \ {q_high_y:.4f}]$" + "\n"
            rf"$R_{{\mathrm{{obs}}, y}}: {r_obs_y:.4f}$"
        )
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.9, edgecolor='#cccccc'),
                fontsize=9.2)
        ax.set_xlabel(r"Reaction Force $R_y$", fontsize=13)
        ax.set_ylabel("Density", fontsize=13)
        ax.set_title(rf"Distilled Model: Reaction Force $R_y$ (Step {step}, Holes)", fontsize=13)
        ax.grid(True, alpha=0.25, linestyle='--')
        ax.legend(loc="upper right", fontsize=8.8, framealpha=0.92, edgecolor='#cccccc')

        # --- ROW 1: GP Ry (Grey) ---
        ax = axes[1]
        ax.hist(ry_gp_eval, bins=35, density=True, alpha=0.35, color='#7f7f7f', edgecolor='#555555', label=r"GP Samples ($R_y$)")
        ax.axvline(r_obs_y, color='red', lw=2.0, linestyle='-', label=rf"Observed $R_{{\mathrm{{obs}}, y}} = {r_obs_y:.4f}$")
        ax.axvline(mu_ry_gp, color='#444444', lw=1.8, linestyle='--', label=rf"GP Mean: {mu_ry_gp:.4f}")
        ax.axvline(q_low_y_gp, color='#888888', lw=1.4, linestyle=':', label=rf"GP $q_{{0.025}}, q_{{0.975}}$")
        ax.axvline(q_high_y_gp, color='#888888', lw=1.4, linestyle=':')
        ax.axvspan(q_low_y_gp, q_high_y_gp, color='#7f7f7f', alpha=0.12, label=r"GP 95% CI")

        stats_text = (
            rf"$\mathbf{{Gaussian \ Process \ R_y \ (Step \ {step})}}$" + "\n"
            rf"$\mathrm{{Mean}}: {mu_ry_gp:.4f}$" + "\n"
            rf"$\mathrm{{Std}}: {format_sci(std_ry_gp)}$" + "\n"
            rf"$95\%\ \mathrm{{CI}}: [{q_low_y_gp:.4f}, \ {q_high_y_gp:.4f}]$" + "\n"
            rf"$R_{{\mathrm{{obs}}, y}}: {r_obs_y:.4f}$"
        )
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.9, edgecolor='#cccccc'),
                fontsize=9.2)
        ax.set_xlabel(r"Reaction Force $R_y$", fontsize=13)
        ax.set_ylabel("Density", fontsize=13)
        ax.set_title(rf"Gaussian Process: Reaction Force $R_y$ (Step {step}, Holes)", fontsize=13)
        ax.grid(True, alpha=0.25, linestyle='--')
        ax.legend(loc="upper right", fontsize=8.8, framealpha=0.92, edgecolor='#cccccc')

    else:
        fig, ax = plt.subplots(figsize=(7.8, 5.5))
        ax.hist(ry_step_eval, bins=35, density=True, alpha=0.35, color='#2ca02c', edgecolor='#2ca02c', label=r"Distilled Samples ($R_y$)")
        ax.axvline(r_obs_y, color='red', lw=2.0, linestyle='-', label=rf"Observed $R_{{\mathrm{{obs}}, y}} = {r_obs_y:.4f}$")
        ax.axvline(mu_ry, color='#2ca02c', lw=1.8, linestyle='--', label=rf"Distilled Mean: {mu_ry:.4f}")
        ax.axvline(q_low_y, color='#74c476', lw=1.4, linestyle=':', label=rf"Distilled $q_{{0.025}}, q_{{0.975}}$")
        ax.axvline(q_high_y, color='#74c476', lw=1.4, linestyle=':')
        ax.axvspan(q_low_y, q_high_y, color='#2ca02c', alpha=0.12, label=r"Distilled 95% CI")
        stats_text = (
            rf"$\mathbf{{Distilled \ R_y \ (Step \ {step})}}$" + "\n"
            rf"$\mathrm{{Mean}}: {mu_ry:.4f}$" + "\n"
            rf"$\mathrm{{Std}}: {format_sci(std_ry)}$" + "\n"
            rf"$95\%\ \mathrm{{CI}}: [{q_low_y:.4f}, \ {q_high_y:.4f}]$" + "\n"
            rf"$R_{{\mathrm{{obs}}, y}}: {r_obs_y:.4f}$"
        )
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.9, edgecolor='#cccccc'), fontsize=9.2)
        ax.set_xlabel(r"Reaction Force $R_y$", fontsize=13)
        ax.set_ylabel("Density", fontsize=13)
        ax.set_title(rf"Reaction Force $R_y$ Distribution (Step {step}, Holes)", fontsize=13)
        ax.grid(True, alpha=0.25, linestyle='--')
        ax.legend(loc="upper right", fontsize=8.8, framealpha=0.92, edgecolor='#cccccc')

    plt.tight_layout()
    os.makedirs(save_path, exist_ok=True)
    out_pdf = os.path.join(save_path, "reaction_force_distribution_holes.pdf")
    save_figure(fig, out_pdf, make_png=make_png)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot reaction force calibration from distilled FEM simulations")
    parser.add_argument("--model_path", type=str, required=True, help="Path to fem_validation/{block,holes} or seed directory")
    parser.add_argument("--gp_dir", type=str, default=None, help="Path to extracted GP directory")
    parser.add_argument("--geometry", type=str, default=None, choices=["block", "holes", "auto"], help="Geometry mode")
    parser.add_argument("--save_path", type=str, default=None, help="Output directory for plots")
    parser.add_argument("--step", type=int, default=16, help="Step index for distribution plot (default: 16)")
    parser.add_argument("--make_png", action="store_true", default=True, help="Save PNG along with PDF")

    args = parser.parse_args()
    model_path = Path(args.model_path)

    # Detect geometry and files
    targets = []
    if (model_path / "fem_distilled_samples.npz").exists():
        geom = args.geometry or ("holes" if "holes" in str(model_path).lower() else "block")
        save_p = args.save_path or str(model_path)
        targets.append((str(model_path / "fem_distilled_samples.npz"), geom, save_p))
    else:
        # Check subdirectories block and holes
        for g in ["block", "holes"]:
            cand = model_path / g / "fem_distilled_samples.npz"
            if cand.exists():
                save_p = args.save_path or str(model_path / g)
                targets.append((str(cand), g, save_p))

    if not targets:
        print(f"Error: Could not locate fem_distilled_samples.npz in {model_path}")
        sys.exit(1)

    for data_file, geom, save_p in targets:
        print(f"--- Computing Reaction Forces for {geom.upper()} from {data_file} ---")
        res = compute_distilled_reaction_forces(data_file)
        res_gp = compute_gp_reaction_forces(data_file, gp_dir=args.gp_dir, step=args.step)
        if geom == "block":
            metrics = plot_reaction_force_block(res, save_p, make_png=args.make_png)
            plot_reaction_force_distribution_block(res, save_p, res_gp=res_gp, step=args.step, make_png=args.make_png)
            print(f"✅ Saved Block Reaction Force Plots: {save_p}/reaction_force_distilled_block.pdf & reaction_force_distribution_block.pdf")
            print(f"   R2_Fx: {metrics['r2_force_x']:.4f}, R2_Fy: {metrics['r2_force_y']:.4f}, EC: {metrics['total_ec']:.1f}%")
        else:
            metrics = plot_reaction_force_holes(res, save_p, make_png=args.make_png)
            plot_reaction_force_distribution_holes(res, save_p, res_gp=res_gp, step=args.step, make_png=args.make_png)
            print(f"✅ Saved Holes Reaction Force Plots: {save_p}/reaction_force_distilled_holes.pdf & reaction_force_distribution_holes.pdf")
            print(f"   R2_Fy: {metrics['r2_force_y']:.4f}, EC_Fy: {metrics['ec_force_y']:.1f}%")


if __name__ == "__main__":
    main()

