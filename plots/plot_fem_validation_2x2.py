"""
2x2 FEM forward-validation figure for one geometry of one seed.

  (a) posterior-mean displacement magnitude at the last validation step, over the clean FEM reference mesh
  (b) nodal RMSE of the posterior-mean displacement against the clean FEM solution, over the validation steps
  (c) calibration curves on the validation steps (free nodes for u): empirical vs. nominal coverage of central
      intervals, posterior vs. clean data and predictive vs. observed data, for displacements and reactions
  (d) reaction forces at every load step: FEM-predicted reactions (each distilled sample solved forward),
      clean reactions of the true material, and the noisy observed reactions (filled: validation, hollow: training)

Observed data exist only for the training geometry (geometry_train). Other geometries (holes) are forward
predictions with the inferred material and are compared with the clean FEM solution only (no predictive band,
observations or predictive coverage). For the training geometry, two coverages are reported on the validation steps:
  * posterior coverage:  clean data inside the 95% interval of the FEM-predicted samples (epistemic only)
  * predictive coverage: observed (noisy) data inside the 95% interval of samples + measurement noise,
                         sigma_u = (sigma_x, sigma_y) inferred by the extraction (recipe disp_noise if absent)
                         at the free nodes, sigma_R = load_noise * |mean predicted R| (load noise is not inferred)

The clean reactions are recomputed from the clean FEM displacement with the true material parameters on the
same mesh (identical to the stored reaction_forces_true of the clean dataset). The predicted reactions use the
FEM-solved displacement of each sample, not the observed displacement.

--experimental emulates a real test on the training geometry: no clean fields or clean reactions are used; (a)/(b)
are against the observed displacement, (c)/(d) show only predictive vs. observed (fem_validation_2x2_<geom>_exp.*).

Usage:
  python3 plots/plot_fem_validation_2x2.py --seed_dir results/<exp>/<seed> [--geometries block holes] [--experimental]
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import yaml
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import matplotlib.ticker as ticker
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1 import make_axes_locatable

jax.config.update("jax_enable_x64", True)

from core.utils import deformation_gradient_element, transformation_jacobian
from plots.plot_reaction_force_distilled import piola_stress_2d
from plots.theme import apply_style

ALPHA = 0.05
COLORS = {"x": "#0072B2", "y": "#D55E00"}


def find_config(seed_dir: Path) -> dict:
    for p in (seed_dir.parent / "config.yaml", seed_dir / "extracted" / "config.json"):
        if p.exists():
            with open(p) as f:
                return yaml.safe_load(f) if p.suffix == ".yaml" else json.load(f)
    raise FileNotFoundError(f"No config.yaml above {seed_dir}")


def true_parameters(cfg: dict):
    mp = cfg.get("material_params", {}) or {}
    if not cfg.get("has_ground_truth", True):
        return None
    from core.material_library import true_full_vector
    p = true_full_vector(mp)
    if p is not None and "mu_params" not in mp:
        p = p[:13]                                               # GMR family: former 13-entry layout
    return p


def make_reaction_fn(node_coords, cells, node_type, stress_mode):
    coords = jnp.asarray(node_coords)
    cells = jnp.asarray(cells)
    ce = coords[cells]
    dA = 0.5 * jnp.abs(jnp.linalg.det(transformation_jacobian(ce)))
    fix_x = jnp.asarray(node_type[:, 1] == 1)
    fix_y = jnp.asarray(node_type[:, 2] == 1)
    n_nodes = coords.shape[0]

    def reaction(u, p):
        F, dNdX = deformation_gradient_element(ce, u[cells])
        P = jax.vmap(lambda f: piola_stress_2d(f, p, stress_mode=stress_mode))(F)
        f_el = jnp.einsum("cij,cnj->cni", P, dNdX) * dA[:, None, None]
        f_int = jnp.zeros((n_nodes, 2)).at[cells.reshape(-1)].add(f_el.reshape(-1, 2))
        # reactions balance the internal forces at the fixed boundary
        return jnp.stack([-jnp.sum(jnp.where(fix_x, f_int[:, 0], 0.0)),
                          -jnp.sum(jnp.where(fix_y, f_int[:, 1], 0.0))])

    # (n_steps, n_nodes, 2), (n_params,) -> (n_steps, 2)
    return jax.jit(jax.vmap(reaction, in_axes=(0, None)))


def interval(samples, axis=0):
    return np.quantile(samples, [ALPHA / 2, 1 - ALPHA / 2], axis=axis)


def coverage(ref, lo, hi):
    return float(np.mean((ref >= lo) & (ref <= hi)) * 100.0)


LEVELS = np.linspace(0.05, 0.99, 48)


def calibration_curve(samples, ref):
    """Empirical coverage of central intervals at the nominal LEVELS (samples on axis 0)."""
    flat = samples.reshape(samples.shape[0], -1)
    q = np.quantile(flat, np.concatenate([(1 - LEVELS) / 2, (1 + LEVELS) / 2]), axis=0)
    lo, hi = q[:len(LEVELS)], q[len(LEVELS):]
    r = np.ravel(ref)[None]
    return np.mean((r >= lo) & (r <= hi), axis=1) * 100.0


def r2(ref, pred):
    ref, pred = np.ravel(ref), np.ravel(pred)
    return float(1.0 - np.sum((ref - pred) ** 2) / np.sum((ref - ref.mean()) ** 2))


def plot_geometry(seed_dir: Path, geom: str, cfg: dict, n_noise_draws: int = 4, batch: int = 64,
                  experimental: bool = False):
    gdir = seed_dir / "fem_validation" / geom
    npz = gdir / "fem_distilled_samples.npz"
    if not npz.exists():
        print(f"[2x2] {npz} missing, skipping {geom}")
        return None
    d = np.load(npz, allow_pickle=True)
    u_pred = np.asarray(d["u_pred"], dtype=float)            # (S, T, N, 2)
    params = np.asarray(d["selected_samples"], dtype=float)  # (S, P)
    coords, cells, node_type = d["node_coords"], d["cells"], np.asarray(d["node_type"])
    u_obs = np.asarray(d["u_exp"], dtype=float)
    has_truth = bool(cfg.get("has_ground_truth", True)) and "u_true" in d.files
    if not experimental and not has_truth:
        print(f"[2x2] {geom}: no ground truth (experimental data); only the --experimental figure applies")
        return None
    u_clean = np.asarray(d["u_true"], dtype=float) if "u_true" in d.files else u_obs
    r_obs = np.asarray(d["loads"], dtype=float)              # observed (noisy) reactions, (T, 2)
    stress_mode = str(d["stress_mode"]) if "stress_mode" in d else "plane_strain"
    S, T = u_pred.shape[:2]

    # failed forward solves are stored as non-finite samples
    ok = np.all(np.isfinite(u_pred.reshape(S, -1)), axis=1)
    u_pred, params = u_pred[ok], params[ok]
    S = u_pred.shape[0]

    val_steps = [int(s) for s in cfg.get("val_load_steps_indices", []) if 0 <= int(s) < T]
    if not val_steps:
        val_steps = list(range(T))
    # kinematic noise: inferred per direction by the extraction (sigma_x, sigma_y); recipe value as fallback
    disp_noise = np.full(2, float(cfg.get("disp_noise", 0.0)))
    noise_source = "recipe"
    em = seed_dir / "extracted" / "extraction_metrics.json"
    if em.exists():
        with open(em) as f:
            m = json.load(f)
        if "sigma_u_x" in m and "sigma_u_y" in m:
            disp_noise = np.array([m["sigma_u_x"], m["sigma_u_y"]], dtype=float)
            noise_source = "inferred"
    load_noise = float(cfg.get("load_noise", 0.0))

    # observed data only exist for the training geometry; other geometries are forward predictions with the
    # inferred material and are compared with the clean FEM solution only
    with_obs = geom == cfg.get("geometry_train", cfg.get("geometry", "block"))
    # experimental mode emulates a real test: no clean fields, the observed data are the only reference
    if experimental and not with_obs:
        print(f"[2x2] experimental mode needs observations; skipping {geom}")
        return None

    free = np.all(node_type[:, 1:5] != 1, axis=1)
    if str(d["control_mode"]) != "displacement":
        free = (node_type[:, 1] != 1) & (node_type[:, 2] != 1)

    # ---------------- displacements on the validation steps, free nodes ----------------
    up = u_pred[:, val_steps][:, :, free]                    # (S, V, Nf, 2)
    uc = u_clean[val_steps][:, free]
    uo = u_obs[val_steps][:, free]
    u_mean = up.mean(0)
    post_lo, post_hi = interval(up)
    uref = uo if experimental else uc
    calib = {} if experimental else {"u_post": calibration_curve(up, uc)}
    ec_u_post = np.nan if experimental else coverage(uc, post_lo, post_hi)
    rng = np.random.default_rng(0)
    if with_obs:
        up_noisy = np.concatenate([up + disp_noise * rng.standard_normal(up.shape) for _ in range(n_noise_draws)])
        pred_lo, pred_hi = interval(up_noisy)
        calib["u_pred"] = calibration_curve(up_noisy, uo)
        ec_u_pred = coverage(uo, pred_lo, pred_hi)
        del up_noisy
    ec_u_post_xy = [coverage(uc[..., i], post_lo[..., i], post_hi[..., i]) for i in range(2)]
    r2_u = [r2(uref[..., i], u_mean[..., i]) for i in range(2)]
    rmse_u = [float(np.sqrt(np.mean((uref[..., i] - u_mean[..., i]) ** 2))) for i in range(2)]

    # ---------------- reactions at every step ----------------
    rfn = make_reaction_fn(coords, cells, node_type, stress_mode)
    r_pred = np.concatenate([
        np.asarray(jax.vmap(rfn, in_axes=(0, 0))(jnp.asarray(u_pred[b:b + batch]), jnp.asarray(params[b:b + batch])))
        for b in range(0, S, batch)])                        # (S, T, 2)
    p_true = true_parameters(cfg)
    r_clean = np.asarray(rfn(jnp.asarray(u_clean), jnp.asarray(p_true))) if (p_true is not None and not experimental) else None
    r_mean = r_pred.mean(0)
    r_post_lo, r_post_hi = interval(r_pred)
    # components with a load cell (holes: only R_y)
    comps = [i for i in range(2) if np.any(np.abs(r_obs[:, i]) > 0)]
    V = np.array(val_steps)
    ec_r_post = coverage(r_clean[V][:, comps], r_post_lo[V][:, comps], r_post_hi[V][:, comps]) if r_clean is not None else np.nan
    r_ref = r_obs if experimental else r_clean
    r2_r = {("xy"[i]): r2(r_ref[V, i], r_mean[V, i]) for i in comps} if r_ref is not None else {}
    # pooled over components (validation steps): displacement on the free nodes, measured reaction components
    r2_u_all = r2(uref, u_mean)
    rmse_u_all = float(np.sqrt(np.mean((uref - u_mean) ** 2)))
    r2_r_all = r2(r_ref[V][:, comps], r_mean[V][:, comps]) if r_ref is not None else np.nan
    rmse_r_all = float(np.sqrt(np.mean((r_ref[V][:, comps] - r_mean[V][:, comps]) ** 2))) if r_ref is not None else np.nan
    if r_clean is not None:
        calib["R_post"] = calibration_curve(r_pred[:, V][:, :, comps], r_clean[V][:, comps])
    if with_obs:
        sig_r = load_noise * np.abs(r_mean)
        r_noisy = np.concatenate([r_pred + sig_r * rng.standard_normal(r_pred.shape) for _ in range(n_noise_draws)])
        r_pred_lo, r_pred_hi = interval(r_noisy)
        ec_r_pred = coverage(r_obs[V][:, comps], r_pred_lo[V][:, comps], r_pred_hi[V][:, comps])
        calib["R_pred"] = calibration_curve(r_noisy[:, V][:, :, comps], r_obs[V][:, comps])
    train_steps = [int(s) for s in cfg.get("train_load_steps_indices", []) if 0 <= int(s) < T]

    # ---------------- figure ----------------
    apply_style()
    fig = plt.figure(figsize=(11.0, 9.6))
    gs = fig.add_gridspec(2, 2, hspace=0.22, wspace=0.28, height_ratios=[1, 1])
    ax_a, ax_b = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    ax_c, ax_d = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])

    # (a) displacement magnitude at the last validation step
    t_f = val_steps[-1]
    u_c_f, u_m_f = (u_obs if experimental else u_clean)[t_f], u_pred[:, t_f].mean(0)
    mag_c, mag_m = np.linalg.norm(u_c_f, axis=1), np.linalg.norm(u_m_f, axis=1)
    vmin, vmax = min(mag_c.min(), mag_m.min()), max(mag_c.max(), mag_m.max())
    tri_c = tri.Triangulation(*(coords + u_c_f).T, cells)
    tri_m = tri.Triangulation(*(coords + u_m_f).T, cells)
    ax_a.tripcolor(tri_c, mag_c, cmap="Blues", alpha=0.30, vmin=vmin, vmax=vmax, zorder=1)
    ax_a.triplot(tri_c, color="#444444", linestyle=":", linewidth=0.7, alpha=0.75, zorder=2)
    im_a = ax_a.tripcolor(tri_m, mag_m, cmap="Blues", alpha=0.88, vmin=vmin, vmax=vmax, zorder=3)
    ax_a.triplot(tri_m, color="#002b4d", linewidth=0.35, alpha=0.45, zorder=4)
    ax_a.set_aspect("equal"); ax_a.axis("off")
    ax_a.set_title(f"(a) Predicted displacement, step {t_f}", fontsize=11)
    cax = make_axes_locatable(ax_a).append_axes("right", size="5%", pad=0.08)
    cb = fig.colorbar(im_a, cax=cax); cb.set_label(r"$\|\bar{\mathbf{u}}_{\mathrm{pred}}\|$")
    cb.locator = ticker.MaxNLocator(nbins=4); cb.update_ticks()
    ref_lbl = r"Observed ($\mathbf{u}_{\mathrm{obs}}$)" if experimental else r"Clean FEM ($\mathbf{u}_{\mathrm{clean}}$)"
    ax_a.legend(handles=[mlines.Line2D([], [], color="#444444", ls=":", label=ref_lbl),
                         mlines.Line2D([], [], color="#002b4d", ls="-", label=r"Posterior mean ($\bar{\mathbf{u}}_{\mathrm{pred}}$)")],
                loc="upper center", bbox_to_anchor=(0.5, -0.01), ncol=2, frameon=False, fontsize=8.5)

    # (b) nodal RMSE over the validation steps
    u_ref_all = u_obs if experimental else u_clean
    err = np.sqrt(np.mean((u_pred[:, val_steps].mean(0) - u_ref_all[val_steps]) ** 2, axis=(0, 2)))
    tri_ref = tri.Triangulation(*coords.T, cells)
    im_b = ax_b.tripcolor(tri_ref, err, cmap="Oranges")
    ax_b.set_aspect("equal"); ax_b.axis("off")
    ax_b.set_title("(b) Nodal RMSE vs. " + ("observed" if experimental else "clean FEM") + ", validation steps", fontsize=11)
    cax = make_axes_locatable(ax_b).append_axes("right", size="5%", pad=0.08)
    cb = fig.colorbar(im_b, cax=cax); cb.set_label(r"$\mathrm{RMSE}$")
    cb.locator = ticker.MaxNLocator(nbins=4); cb.update_ticks()

    # (c) calibration: empirical vs nominal coverage of central intervals, validation steps
    lv = LEVELS * 100
    ax_c.plot([0, 100], [0, 100], "k:", lw=1.0, label="Ideal")
    if "u_post" in calib:
        ax_c.plot(lv, calib["u_post"], "-", color="#1b4f72", lw=2.0, label=r"$\mathbf{u}$: posterior vs. clean")
    if "u_pred" in calib:
        ax_c.plot(lv, calib["u_pred"], "--", color="#1b4f72", lw=2.0, label=r"$\mathbf{u}$: predictive vs. observed")
    if "R_post" in calib:
        ax_c.plot(lv, calib["R_post"], "-", color="#b9770e", lw=1.5, marker="o", ms=2.5, label=r"$R$: posterior vs. clean")
    if "R_pred" in calib:
        ax_c.plot(lv, calib["R_pred"], "--", color="#b9770e", lw=1.5, marker="o", ms=2.5, label=r"$R$: predictive vs. observed")
    ax_c.axvline(95, color="gray", lw=0.8, alpha=0.6)
    ax_c.set_xlim(0, 100); ax_c.set_ylim(0, 102)
    ax_c.set_xlabel("Nominal coverage of central interval (%)"); ax_c.set_ylabel("Empirical coverage (%)")
    ax_c.grid(True, ls=":", alpha=0.6); ax_c.set_box_aspect(1)
    ax_c.legend(loc="lower right", fontsize=8.3, framealpha=0.92)
    ax_c.set_title("(c) Calibration, validation steps (free nodes)", fontsize=11)

    # (d) reactions, all steps; filled markers on validation steps, hollow on training steps
    steps = np.arange(T)
    is_val = np.isin(steps, val_steps)
    is_train = np.isin(steps, train_steps) & ~is_val
    for i in comps:
        c = "xy"[i]; col = COLORS[c]
        if with_obs:
            ax_d.fill_between(steps, r_pred_lo[:, i], r_pred_hi[:, i], color=col, alpha=0.13, lw=0, zorder=1)
        ax_d.fill_between(steps, r_post_lo[:, i], r_post_hi[:, i], color=col, alpha=0.35, lw=0, zorder=2)
        ax_d.plot(steps, r_mean[:, i], "--", color=col, lw=1.4, zorder=3)
        if r_clean is not None:
            ax_d.plot(steps, r_clean[:, i], "-", color="k", lw=1.1, zorder=4)
            if not with_obs:
                ax_d.plot(steps[is_val], r_clean[is_val, i], "o", ms=4.0, color=col, mec="k", mew=0.6, zorder=6)
        if with_obs:
            ax_d.plot(steps[is_val], r_obs[is_val, i], "s", ms=4.5, color=col, mec="k", mew=0.6, zorder=6)
            ax_d.plot(steps[is_train], r_obs[is_train, i], "s", ms=4.0, mfc="white", mec=col, mew=1.0, zorder=5)
        ax_d.annotate(rf"$R_{c}$", (steps[-1], r_mean[-1, i]), xytext=(4, 0), textcoords="offset points",
                      color=col, fontsize=10, va="center")
    ax_d.set_xlabel("Load step"); ax_d.set_ylabel("Reaction force (N)")
    ax_d.set_xlim(-0.5, T - 0.5)
    ax_d.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=10))
    ax_d.grid(True, ls=":", alpha=0.5)
    ax_d.set_title("(d) Reaction forces, all steps", fontsize=11)
    handles = [mlines.Line2D([], [], color="k", lw=1.1, label=r"Clean $R_{\mathrm{clean}}$ (true material)"),
               mlines.Line2D([], [], color="gray", ls="--", lw=1.4, label="Posterior mean (FEM)"),
               mpatches.Patch(color="gray", alpha=0.45, label="Posterior 95% (epistemic)")]
    if with_obs:
        handles += [mpatches.Patch(color="gray", alpha=0.18, label="Predictive 95% (+ load noise)"),
                    mlines.Line2D([], [], color="gray", marker="s", mec="k", ls="", label=r"Observed $R_{\mathrm{obs}}$, validation step"),
                    mlines.Line2D([], [], color="gray", marker="s", mfc="white", ls="", label=r"Observed $R_{\mathrm{obs}}$, training step")]
    elif r_clean is not None:
        handles += [mlines.Line2D([], [], color="gray", marker="o", mec="k", ls="", label=r"$R_{\mathrm{clean}}$, validation step")]
    if r_clean is None:
        handles = handles[1:]
    ax_d.legend(handles=handles, loc="upper left", fontsize=7.8, framealpha=0.92)

    # metrics banner
    r2r = ", ".join(rf"$r^2_{{R_{c}}} = {v:.4f}$" for c, v in r2_r.items())
    if experimental:
        line1 = (rf"Predictive vs. observed (val. steps):  $\mathrm{{EC}}_u = \mathbf{{{ec_u_pred:.1f}\%}}$,  "
                 rf"$\mathrm{{EC}}_R = \mathbf{{{ec_r_pred:.1f}\%}}$   $\vert$   "
                 rf"vs. observed: $r^2_{{u_x}} = {r2_u[0]:.4f},\ r^2_{{u_y}} = {r2_u[1]:.4f}$, {r2r}")
        line2 = (rf"Emulated experiment (observed data only)   $\vert$   "
                 rf"$\mathrm{{RMSE}}_{{u_x}} = {rmse_u[0]:.2e},\ \mathrm{{RMSE}}_{{u_y}} = {rmse_u[1]:.2e}$ "
                 rf"(noise $\hat\sigma_x = {disp_noise[0]:.2e},\ \hat\sigma_y = {disp_noise[1]:.2e}$, {noise_source})   $\vert$   "
                 rf"$\sigma_R = {load_noise:g}\,|\bar R|$   $\vert$   {len(val_steps)} val. steps, {S} samples")
    line1 = line1 if experimental else (rf"Posterior vs. clean (val. steps):  $\mathrm{{EC}}_u = \mathbf{{{ec_u_post:.1f}\%}}$,  "
             rf"$\mathrm{{EC}}_R = \mathbf{{{ec_r_post:.1f}\%}}$   $\vert$   "
             rf"$r^2_{{u_x}} = {r2_u[0]:.4f},\ r^2_{{u_y}} = {r2_u[1]:.4f}$   $\vert$   {r2r}")
    if experimental:
        pass
    elif with_obs:
        line2 = (rf"Predictive vs. observed (val. steps):  $\mathrm{{EC}}_u = \mathbf{{{ec_u_pred:.1f}\%}}$,  "
                 rf"$\mathrm{{EC}}_R = \mathbf{{{ec_r_pred:.1f}\%}}$   $\vert$   "
                 rf"$\hat\sigma_x = {disp_noise[0]:.2e},\ \hat\sigma_y = {disp_noise[1]:.2e}$ ({noise_source}),  "
                 rf"$\sigma_R = {load_noise:g}\,|\bar R|$   $\vert$   "
                 rf"{len(val_steps)} val. steps, {S} samples, nominal 95%")
    else:
        line2 = (rf"Forward prediction with the inferred material (no observations of this geometry)   $\vert$   "
                 rf"{len(val_steps)} val. steps, {S} samples, nominal 95%")
    fig.text(0.5, 0.0, line1 + "\n" + line2, ha="center", va="center", fontsize=9.2, linespacing=1.6,
             bbox=dict(boxstyle="round,pad=0.45", facecolor="#f5f5f5", edgecolor="#bbbbbb", lw=0.6))
    fig.suptitle(f"FEM forward validation, {geom} geometry" + (" (emulated experiment)" if experimental else ""),
                 fontsize=12.5, y=0.985)
    plt.subplots_adjust(left=0.04, right=0.96, top=0.94, bottom=0.12)

    tag = f"{geom}_exp" if experimental else geom
    out = gdir / f"fem_validation_2x2_{tag}.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)

    if experimental:
        metrics = {"geometry": geom, "mode": "experimental", "val_steps": val_steps, "n_samples": int(S),
                   "disp_noise": disp_noise.tolist(), "disp_noise_source": noise_source, "load_noise": load_noise,
                   "reaction_components": ["xy"[i] for i in comps],
                   "predictive_obs": {"ec_u": ec_u_pred, "ec_R": ec_r_pred, "r2_ux": r2_u[0], "r2_uy": r2_u[1],
                                      "rmse_ux": rmse_u[0], "rmse_uy": rmse_u[1], **{f"r2_R{c}": v for c, v in r2_r.items()}},
                   "calibration": {"nominal": (LEVELS * 100).tolist(), **{k: v.tolist() for k, v in calib.items()}}}
        with open(gdir / f"fem_validation_2x2_{tag}.json", "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"[2x2] {tag}: predictive EC_u={ec_u_pred:.1f}% EC_R={ec_r_pred:.1f}% -> {out}")
        return metrics
    metrics = {
        "geometry": geom, "val_steps": val_steps, "n_samples": int(S), "n_failed_samples": int((~ok).sum()),
        "disp_noise": disp_noise.tolist(), "disp_noise_source": noise_source, "load_noise": load_noise, "reaction_components": ["xy"[i] for i in comps],
        "posterior_clean": {"ec_u": ec_u_post, "ec_ux": ec_u_post_xy[0], "ec_uy": ec_u_post_xy[1], "ec_R": ec_r_post,
                            "r2_u": r2_u_all, "rmse_u": rmse_u_all, "r2_R": r2_r_all, "rmse_R": rmse_r_all,
                            "r2_ux": r2_u[0], "r2_uy": r2_u[1], "rmse_ux": rmse_u[0], "rmse_uy": rmse_u[1],
                            **{f"r2_R{c}": v for c, v in r2_r.items()}},
        "calibration": {"nominal": (LEVELS * 100).tolist(), **{k: v.tolist() for k, v in calib.items()}},
    }
    if with_obs:
        metrics["predictive_obs"] = {"ec_u": ec_u_pred, "ec_R": ec_r_pred}
    else:
        for k in ("disp_noise", "disp_noise_source", "load_noise"):
            metrics.pop(k)
    with open(gdir / f"fem_validation_2x2_{geom}.json", "w") as f:
        json.dump(metrics, f, indent=2)
    pred_str = f" | predictive EC_u={ec_u_pred:.1f}% EC_R={ec_r_pred:.1f}%" if with_obs else ""
    print(f"[2x2] {geom}: posterior EC_u={ec_u_post:.1f}% EC_R={ec_r_post:.1f}%{pred_str} -> {out}")
    return metrics


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed_dir", required=True)
    ap.add_argument("--geometries", nargs="+", default=["block", "holes"])
    ap.add_argument("--experimental", action="store_true",
                    help="emulate a real test: observed data only (training geometry), writes *_exp.pdf/json")
    args = ap.parse_args()
    seed_dir = Path(args.seed_dir).resolve()
    cfg = find_config(seed_dir)
    results = {(f"{g}_exp" if args.experimental else g): m for g in args.geometries
               if (m := plot_geometry(seed_dir, g, cfg, experimental=args.experimental)) is not None}
    # the coverages also go into the seed's validation_metrics.json files (key "fem_2x2")
    for p in (seed_dir / "distilled" / "validation_metrics.json", seed_dir / "fem_validation" / "validation_metrics.json",
              seed_dir / "validation_metrics.json"):
        if results and p.exists():
            with open(p) as f:
                vm = json.load(f)
            vm.setdefault("fem_2x2", {}).update(results)
            with open(p, "w") as f:
                json.dump(vm, f, indent=1)


if __name__ == "__main__":
    main()
