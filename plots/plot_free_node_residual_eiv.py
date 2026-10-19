"""
Pre-FEM free-node check of the distilled material posterior (companion of the reaction-force breakdown in
distilled/block_validation).

Parameter samples of the distilled posterior are passed through Psi and the FE-VFM operator at the observed
displacement field of every load step, as for the reactions. At the free degrees of freedom this gives

  * the internal-force residual f = G_f(u_obs; theta), which vanishes only at equilibrium, and
  * the errors-in-variables correction eps = K(theta)^-1 f, the one Newton step from the observed state to
    equilibrium. Under the EIV likelihood it is distributed as the displacement noise, eps ~ N(0, sigma_u^2),
    so it is compared with the noise level inferred by the extraction (sigma_x, sigma_y).

The true material evaluated on the same observed field is the reference: its residual is produced by the
measurement noise alone (on the clean field it is the FE equilibrium error, ~0).

Figure (free_node_residual_eiv_block.pdf):
  (a) RMS free-node force residual per load step: posterior 95% band and median, true material on u_obs / u_clean
  (b) RMS of the EIV correction per load step and direction, with the inferred noise levels
  (c) posterior-mean |eps| / sigma at the free nodes, last validation step
  (d) standardized correction eps / sigma on the validation steps (all samples) against N(0, 1)

Usage:
  python3 plots/plot_free_node_residual_eiv.py --seed_dir results/<exp>/<seed> [--n_samples 1024]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import yaml
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable

jax.config.update("jax_enable_x64", True)

from core.dataset_store import load_dataset
from core.utils import deformation_gradient_element, transformation_jacobian
from plots.plot_reaction_force_distilled import piola_stress_2d
from plots.theme import apply_style

COLORS = {"x": "#0072B2", "y": "#D55E00"}


def build_operators(mesh_pos, cells, stress_mode):
    """Element internal forces and element stiffnesses (by forward differentiation) for a parameter vector."""
    ce = jnp.asarray(mesh_pos)[jnp.asarray(cells)]
    _, dNdX = deformation_gradient_element(ce, jnp.zeros(ce.shape))          # (C, 3, 2)
    dA = 0.5 * jnp.abs(jnp.linalg.det(transformation_jacobian(ce)))         # (C,)

    def f_elem(u_e, dN, da, p):                                             # (3, 2) -> (3, 2)
        F = jnp.eye(2) + jnp.einsum("ai,aj->ij", u_e, dN)
        P = piola_stress_2d(F, p, stress_mode=stress_mode)
        return jnp.einsum("ij,aj->ai", P, dN) * da

    k_elem = jax.jacfwd(f_elem)                                             # (3, 2, 3, 2)
    return dNdX, dA, f_elem, k_elem


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed_dir", required=True)
    ap.add_argument("--n_samples", type=int, default=1024, help="posterior samples for the force residual")
    ap.add_argument("--n_eps_samples", type=int, default=256, help="of these, samples for the EIV correction (K solves)")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--save_path", default=None, help="default: <seed>/distilled/block_validation")
    ap.add_argument("--use_fem_bundle", action="store_true",
                    help="take the training kinematics from <seed>/fem_validation/block/fem_distilled_samples.npz")
    ap.add_argument("--detail_step", type=int, nargs="*", default=[10, 18],
                    help="load steps of the per-node figures (none: skip)")
    args = ap.parse_args()

    seed_dir = Path(args.seed_dir).resolve()
    ddir, edir = seed_dir / "distilled", seed_dir / "extracted"
    out_dir = Path(args.save_path) if args.save_path else ddir / "block_validation"
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(open(seed_dir.parent / "config.yaml"))
    ecfg = json.load(open(edir / "config.json"))
    em = json.load(open(edir / "extraction_metrics.json"))

    bundle = seed_dir / "fem_validation" / "block" / "fem_distilled_samples.npz"
    if args.use_fem_bundle and bundle.exists():
        from core.dataset_store import dataset_from_fem_bundle
        data = dataset_from_fem_bundle(str(bundle))
    else:
        spec = open(seed_dir / "dataset_train.spec").read().strip()
        data = load_dataset(spec)
    mesh_pos, cells, node_type = np.asarray(data["mesh_pos"]), np.asarray(data["cells"]), np.asarray(data["node_type"])
    u_obs, u_clean = np.asarray(data["u_obs"], float), np.asarray(data["u_true"], float)
    stress_mode = str(data["stress_mode"])
    T, N = u_obs.shape[:2]
    free = np.all(node_type[:, 1:5] != 1, axis=1)
    if str(data["control_mode"]) != "displacement":
        free = (node_type[:, 1] != 1) & (node_type[:, 2] != 1)
    free_dofs = np.stack([np.where(free)[0] * 2, np.where(free)[0] * 2 + 1], axis=1).reshape(-1)  # x, y interleaved
    comp_of_dof = np.tile([0, 1], free.sum())

    val_steps = [int(s) for s in cfg.get("val_load_steps_indices", []) if 0 <= int(s) < T]
    train_steps = [int(s) for s in ecfg.get("train_load_steps_indices", []) if 0 <= int(s) < T]
    sig = np.array([em["sigma_u_x"], em["sigma_u_y"]], float)

    # posterior samples: dev and vol are distilled separately, drawn independently
    rng = np.random.default_rng(0)
    dev, vol = np.load(ddir / "dev_flow_samples.npy"), np.load(ddir / "vol_flow_samples.npy")
    S = min(args.n_samples, len(dev), len(vol))
    params = np.concatenate([dev[rng.choice(len(dev), S, replace=False)], vol[rng.choice(len(vol), S, replace=False)]], 1)
    mp = cfg.get("material_params", {}) or {}
    p_true = np.array(list(mp["dev_params"]) + list(mp["vol_params"]), float) if "dev_params" in mp else None

    dNdX, dA, f_elem, k_elem = build_operators(mesh_pos, cells, stress_mode)
    cells_j = jnp.asarray(cells)
    elem_dofs = jnp.stack([cells_j * 2, cells_j * 2 + 1], axis=-1).reshape(-1, 6)   # (C, 6)

    @jax.jit
    def residual(u, p):                                            # (N, 2) -> free-node force residual (nf,)
        fe = jax.vmap(f_elem, in_axes=(0, 0, 0, None))(u[cells_j], dNdX, dA, p)
        f = jnp.zeros(2 * N).at[elem_dofs.reshape(-1)].add(fe.reshape(-1))
        return f[free_dofs]

    @jax.jit
    def eiv_correction(u, p):                                      # eps = K_ff^-1 f_f and lambda_min of K_ff
        fe = jax.vmap(f_elem, in_axes=(0, 0, 0, None))(u[cells_j], dNdX, dA, p)
        ke = jax.vmap(k_elem, in_axes=(0, 0, 0, None))(u[cells_j], dNdX, dA, p).reshape(-1, 6, 6)
        f = jnp.zeros(2 * N).at[elem_dofs.reshape(-1)].add(fe.reshape(-1))
        K = jnp.zeros((2 * N, 2 * N)).at[elem_dofs[:, :, None], elem_dofs[:, None, :]].add(ke)
        Kff = K[free_dofs][:, free_dofs]
        return jnp.linalg.solve(Kff, f[free_dofs]), jnp.linalg.eigvalsh(0.5 * (Kff + Kff.T))[0]

    res_steps = jax.vmap(residual, in_axes=(0, None))              # over steps
    eps_steps = jax.vmap(eiv_correction, in_axes=(0, None))
    U_obs, U_clean = jnp.asarray(u_obs), jnp.asarray(u_clean)

    # force residual, all samples and steps: RMS over free dofs -> (S, T)
    f_rms = np.concatenate([np.sqrt(np.mean(np.asarray(jax.vmap(res_steps, in_axes=(None, 0))(U_obs, jnp.asarray(params[b:b + args.batch]))) ** 2, -1))
                            for b in range(0, S, args.batch)])
    # EIV correction on a subset: (Se, T, nf), lambda_min (Se, T)
    Se = min(args.n_eps_samples, S)
    eps, lmin = [], []
    for b in range(0, Se, args.batch):
        e, l = jax.vmap(eps_steps, in_axes=(None, 0))(U_obs, jnp.asarray(params[b:min(b + args.batch, Se)]))
        eps.append(np.asarray(e)); lmin.append(np.asarray(l))
    eps, lmin = np.concatenate(eps), np.concatenate(lmin)
    stable = np.all(lmin > 0, axis=1)                              # samples with a positive definite K_ff at every step
    # the correction is only meaningful where K_ff is positive definite: unstable (sample, step) pairs are masked
    eps = np.where((lmin > 0)[:, :, None], eps, np.nan)

    ref = {}
    if p_true is not None:
        pt = jnp.asarray(p_true)
        ref["f_obs"] = np.sqrt(np.mean(np.asarray(res_steps(U_obs, pt)) ** 2, -1))
        ref["f_clean"] = np.sqrt(np.mean(np.asarray(res_steps(U_clean, pt)) ** 2, -1))
        e_true, _ = eps_steps(U_obs, pt)
        ref["eps_obs"] = np.asarray(e_true)
    noise_true = float(cfg.get("disp_noise", np.nan))

    def eps_rms(e, c):                                              # RMS over the free dofs of one direction
        return np.sqrt(np.mean(e[..., comp_of_dof == c] ** 2, axis=-1))   # NaN for masked (unstable) pairs

    # ---------------- figure ----------------
    apply_style()
    fig = plt.figure(figsize=(11.0, 9.6))
    gs = fig.add_gridspec(2, 2, hspace=0.26, wspace=0.28)
    ax_a, ax_b = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    ax_c, ax_d = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])
    steps = np.arange(T)
    is_val = np.isin(steps, val_steps)
    is_train = np.isin(steps, train_steps) & ~is_val

    # (a) force residual
    lo, med, hi = np.quantile(f_rms, [0.025, 0.5, 0.975], axis=0)
    ax_a.fill_between(steps, lo, hi, color="#1b4f72", alpha=0.25, lw=0)
    ax_a.plot(steps, med, "-", color="#1b4f72", lw=1.6)
    ax_a.plot(steps[is_val], med[is_val], "o", color="#1b4f72", ms=4.5, mec="k", mew=0.6)
    ax_a.plot(steps[is_train], med[is_train], "o", color="#1b4f72", ms=4.0, mfc="white", mew=1.0)
    h = [mpatches.Patch(color="#1b4f72", alpha=0.35, label="Posterior 95% / median"),
         mlines.Line2D([], [], color="#1b4f72", marker="o", mec="k", ls="", label="validation step"),
         mlines.Line2D([], [], color="#1b4f72", marker="o", mfc="white", ls="", label="training step")]
    if ref:
        ax_a.plot(steps, ref["f_obs"], "k--", lw=1.2)
        ax_a.plot(steps, ref["f_clean"], ":", color="gray", lw=1.2)
        h += [mlines.Line2D([], [], color="k", ls="--", label=r"True material on $\mathbf{u}_{\mathrm{obs}}$ (noise only)"),
              mlines.Line2D([], [], color="gray", ls=":", label=r"True material on $\mathbf{u}_{\mathrm{clean}}$")]
    ax_a.set_xlabel("Load step"); ax_a.set_ylabel(r"RMS free-node residual $\|\mathbf{f}\|_{\mathrm{rms}}$ (N)")
    ax_a.set_title("(a) Free-node force residual on observed displacements", fontsize=11)
    ax_a.legend(handles=h, loc="upper left", fontsize=7.8, framealpha=0.92)
    ax_a.grid(True, ls=":", alpha=0.5); ax_a.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=10))

    # (b) EIV correction vs inferred noise
    es = eps
    for c, name in enumerate("xy"):
        r = eps_rms(es, c)                                          # (Se, T)
        lo, med, hi = np.nanquantile(r, [0.025, 0.5, 0.975], axis=0)
        col = COLORS[name]
        ax_b.fill_between(steps, lo, hi, color=col, alpha=0.22, lw=0)
        ax_b.plot(steps, med, "-", color=col, lw=1.6, label=rf"Posterior, $\varepsilon_{name}$")
        ax_b.plot(steps[is_val], med[is_val], "o", color=col, ms=4.5, mec="k", mew=0.6)
        ax_b.plot(steps[is_train], med[is_train], "o", color=col, ms=4.0, mfc="white", mew=1.0)
        if ref:
            ax_b.plot(steps, eps_rms(ref["eps_obs"], c), "--", color=col, lw=1.0, alpha=0.9,
                      label=rf"True material, $\varepsilon_{name}$")
        ax_b.axhline(sig[c], color=col, ls=":", lw=1.3, label=rf"Inferred $\hat\sigma_{name}$")
    if np.isfinite(noise_true):
        ax_b.axhline(noise_true, color="k", ls="-", lw=0.8, alpha=0.6, label=r"True $\sigma_u$")
    ax_b.set_yscale("log")
    ax_b.set_xlabel("Load step"); ax_b.set_ylabel(r"RMS EIV correction $\|\boldsymbol{\varepsilon}\|_{\mathrm{rms}}$")
    ax_b.set_title(r"(b) EIV correction $\boldsymbol{\varepsilon} = \mathbf{K}^{-1}\mathbf{f}$ vs. noise level", fontsize=11)
    ax_b.legend(loc="upper left", fontsize=7.4, ncol=2, framealpha=0.92)
    ax_b.grid(True, which="both", ls=":", alpha=0.5); ax_b.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=10))

    # (c) spatial map, last validation step: posterior-mean |eps| / sigma per free node
    t_f = val_steps[-1] if val_steps else T - 1
    e_mean = np.nanmean(es[:, t_f], 0).reshape(-1, 2) / sig[None, :]
    mag = np.full(N, np.nan); mag[free] = np.linalg.norm(e_mean, axis=1)
    node_vals = np.nan_to_num(mag, nan=0.0)
    trian = tri.Triangulation(*(mesh_pos + u_obs[t_f]).T, cells)
    im = ax_c.tripcolor(trian, node_vals, cmap="Purples", shading="gouraud")
    ax_c.triplot(trian, color="#555555", lw=0.25, alpha=0.4)
    ax_c.set_aspect("equal"); ax_c.axis("off")
    ax_c.set_title(rf"(c) Posterior-mean $|\boldsymbol{{\varepsilon}}|/\hat\sigma$ at the free nodes, step {t_f}", fontsize=11)
    cax = make_axes_locatable(ax_c).append_axes("right", size="5%", pad=0.08)
    cb = fig.colorbar(im, cax=cax); cb.set_label(r"$|\bar{\boldsymbol{\varepsilon}}| / \hat\sigma$")

    # (d) standardized correction on the validation steps
    V = np.array(val_steps) if val_steps else steps
    frac_st = np.mean(lmin > 0, axis=0)
    V = V[frac_st[V] >= 0.5] if np.any(frac_st[V] >= 0.5) else V       # steps where the posterior is mostly stable
    z = (es[:, V] / sig[comp_of_dof][None, None, :]).reshape(-1)
    z = z[np.isfinite(z)]
    zr = np.linspace(-5, 5, 400)
    ax_d.hist(np.clip(z, -5, 5), bins=80, density=True, color="#7d3c98", alpha=0.45, label="Posterior samples")
    if ref:
        zt = (ref["eps_obs"][V] / sig[comp_of_dof][None, :]).reshape(-1)
        ax_d.hist(np.clip(zt, -5, 5), bins=80, density=True, histtype="step", color="k", lw=1.2, label="True material")
    ax_d.plot(zr, np.exp(-zr ** 2 / 2) / np.sqrt(2 * np.pi), "--", color="#b9770e", lw=1.6, label=r"$\mathcal{N}(0, 1)$")
    within = float(np.mean(np.abs(z) <= 1.96) * 100)
    std_z = float(np.std(z))
    ax_d.set_xlim(-5, 5)
    ax_d.set_xlabel(r"Standardized correction $\varepsilon / \hat\sigma$"); ax_d.set_ylabel("Density")
    ax_d.set_title("(d) Standardized EIV correction, validation steps", fontsize=11)
    ax_d.legend(loc="upper right", fontsize=8.3, framealpha=0.92)
    ax_d.grid(True, ls=":", alpha=0.5)

    # banner
    vrms = [float(np.nanmedian(eps_rms(es[:, V], c))) for c in range(2)]
    frac_unstable_val = float(np.mean(lmin[:, V] <= 0) * 100)
    t_rms = [float(np.sqrt(np.mean(eps_rms(ref["eps_obs"][V], c) ** 2))) for c in range(2)] if ref else [np.nan, np.nan]
    line1 = (rf"Validation steps: median RMS $\varepsilon_x = {vrms[0]:.2e}$, $\varepsilon_y = {vrms[1]:.2e}$   $\vert$   "
             rf"inferred $\hat\sigma_x = {sig[0]:.2e}$, $\hat\sigma_y = {sig[1]:.2e}$   $\vert$   "
             rf"true material: $\varepsilon_x = {t_rms[0]:.2e}$, $\varepsilon_y = {t_rms[1]:.2e}$")
    line2 = (rf"$|\varepsilon/\hat\sigma| \leq 1.96$: {within:.1f}% (nominal 95%),  std$(\varepsilon/\hat\sigma) = {std_z:.2f}$   $\vert$   "
             rf"{S} samples (force), {Se} samples (EIV; unstable $\mathbf{{K}}_{{ff}}$ in {frac_unstable_val:.1f}% of sample-step pairs on val. steps, excluded)")
    fig.text(0.5, 0.0, line1 + "\n" + line2, ha="center", va="center", fontsize=9.0, linespacing=1.6,
             bbox=dict(boxstyle="round,pad=0.45", facecolor="#f5f5f5", edgecolor="#bbbbbb", lw=0.6))
    fig.suptitle("Pre-FEM free-node check of the distilled posterior on the observed displacements (block)",
                 fontsize=12.5, y=0.985)
    plt.subplots_adjust(left=0.06, right=0.96, top=0.94, bottom=0.12)
    out = out_dir / "free_node_residual_eiv_block.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)

    metrics = {"val_steps": val_steps, "n_samples": S, "n_eps_samples": Se, "n_stable": int(stable.sum()),
               "pct_unstable_pairs_val": frac_unstable_val, "val_steps_used": [int(v) for v in V],
               "pct_unstable_by_step": (np.mean(lmin <= 0, axis=0) * 100).tolist(),
               "sigma_inferred": sig.tolist(), "eps_rms_val_median": vrms, "eps_rms_val_true_material": t_rms,
               "frac_within_1.96": within, "std_standardized": std_z,
               "f_rms_median": np.median(f_rms, 0).tolist(),
               "f_rms_true_obs": ref["f_obs"].tolist() if ref else None}
    json.dump(metrics, open(out_dir / "free_node_residual_eiv_block.json", "w"), indent=2)
    print(f"[free-node] {out}: within 1.96 sigma {within:.1f}%, std {std_z:.2f}, "
          f"median RMS eps (val) x {vrms[0]:.2e} y {vrms[1]:.2e} vs sigma {sig[0]:.2e} {sig[1]:.2e}")

    e_true = (u_obs - u_clean).reshape(T, -1)[:, free_dofs] if p_true is not None else None
    for step in args.detail_step:
        if 0 <= step < T:
            plot_step_detail(es, sig, comp_of_dof, free, mesh_pos, cells, u_obs, step, val_steps,
                             e_true, out_dir, rng)


LEVELS = np.linspace(0.05, 0.99, 48)


def interval_hits(samples, ref, levels=LEVELS):
    """For each nominal level: fraction of entries whose ref lies in the central interval of samples (axis 0)."""
    q = np.nanquantile(samples, np.concatenate([(1 - levels) / 2, (1 + levels) / 2]), axis=0)
    lo, hi = q[:len(levels)], q[len(levels):]
    return (ref[None] >= lo) & (ref[None] <= hi)                       # (L, ...)


def plot_step_detail(eps, sig, comp_of_dof, free, mesh_pos, cells, u_obs, step, val_steps, e_true, out_dir, rng,
                     n_noise=8):
    """
    EIV correction at every free DOF of one load step, and coverage that needs only the observations:
    the predictive draw of the observation is y - eps_s + sigma z, so y_obs lies in its central interval exactly
    when 0 lies in the central interval of eps_s + sigma z. With synthetic data, the latent-state check is whether
    the true noise e = y - u_clean lies in the interval of eps_s (u_clean in the band of y - eps_s).
    """
    sig_dof = sig[comp_of_dof]
    pred = np.concatenate([eps + sig_dof * rng.standard_normal(eps.shape) for _ in range(n_noise)])   # (S*n, T, D)
    i95 = int(np.argmin(np.abs(LEVELS - 0.95)))

    # coverage curves on the validation steps where at least half of the samples have a stable K_ff (elsewhere the
    # posterior has lost stability and the correction is not a noise estimate)
    frac_stable = np.mean(np.isfinite(eps[:, :, 0]), axis=0)             # (T,)
    V_all = [t for t in val_steps if t < eps.shape[1]]
    V = [t for t in V_all if frac_stable[t] >= 0.5]
    V_excl = [t for t in V_all if t not in V]
    if not V:
        print(f"[free-node] no validation step with a stable posterior; skipping step {step}")
        return
    hits_pred_val = interval_hits(pred[:, V], np.zeros(eps.shape[1:])[V])
    cal_pred = hits_pred_val.mean(axis=(1, 2)) * 100
    cal_lat = interval_hits(eps[:, V], e_true[V]).mean(axis=(1, 2)) * 100 if e_true is not None else None

    # the detail step
    e_s = eps[:, step]                                                   # (S, D)
    p_lo, p_hi = np.nanquantile(e_s, [0.025, 0.975], axis=0)
    q_lo, q_hi = np.nanquantile(pred[:, step], [0.025, 0.975], axis=0)
    miss_pred = ~((0 >= q_lo) & (0 <= q_hi))
    ec_pred_step = 100 * (1 - miss_pred.mean())
    ec_lat_step = 100 * np.mean((e_true[step] >= p_lo) & (e_true[step] <= p_hi)) if e_true is not None else np.nan

    free_idx = np.where(free)[0]
    order = np.lexsort((mesh_pos[free_idx, 0], mesh_pos[free_idx, 1]))   # free nodes from bottom to top, then left to right

    apply_style()
    fig = plt.figure(figsize=(12.0, 10.4))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1.25], hspace=0.38, wspace=0.25)
    for r, (c, name) in enumerate([(0, "x"), (1, "y")]):
        ax = fig.add_subplot(gs[r, :])
        sel = np.where(comp_of_dof == c)[0][order]
        k = np.arange(len(sel))
        col = COLORS[name]
        ax.fill_between(k, q_lo[sel], q_hi[sel], step="mid", color=col, alpha=0.15, lw=0, label=r"Predictive 95% ($\varepsilon_s + \hat\sigma z$)")
        ax.fill_between(k, p_lo[sel], p_hi[sel], step="mid", color=col, alpha=0.45, lw=0, label=r"Posterior 95% of $\varepsilon_s$")
        ax.plot(k, np.nanmean(e_s[:, sel], 0), "-", color=col, lw=0.8, label=r"Posterior mean $\bar\varepsilon$")
        if e_true is not None:
            ax.plot(k, e_true[step, sel], ".", color="k", ms=2.5, label=r"True noise $e = y - u_{\mathrm{clean}}$")
        ax.axhline(0, color="gray", lw=0.6)
        for s_ in (-1.96, 1.96):
            ax.axhline(s_ * sig[c], color=col, ls=":", lw=1.0)
        m = miss_pred[sel]
        ax.plot(k[m], np.zeros(m.sum()), "x", color="#c0392b", ms=4, mew=1.2,
                label=r"$y_{\mathrm{obs}}$ outside predictive 95%")
        ax.set_xlim(-1, len(sel)); ax.set_ylabel(rf"$\varepsilon_{name}$")
        ax.set_xlabel("Free node (ordered bottom to top)")
        ax.grid(True, ls=":", alpha=0.4)
        ax.set_title(rf"({'ab'[r]}) EIV correction $\varepsilon_{name} = [\mathbf{{K}}^{{-1}}\mathbf{{f}}]_{name}$ at every free node, step {step}"
                     rf" (dotted: $\pm1.96\hat\sigma_{name}$)", fontsize=11)
        if r == 0:
            ax.legend(loc="upper right", fontsize=7.8, ncol=3, framealpha=0.92)

    # (c) map: posterior-mean |eps|/sigma, with nodes whose observation falls outside the predictive interval
    ax_c = fig.add_subplot(gs[2, 0])
    N = mesh_pos.shape[0]
    z_mean = np.linalg.norm((np.nanmean(e_s, 0) / sig_dof).reshape(-1, 2), axis=1)
    vals = np.zeros(N); vals[free] = z_mean
    trian = tri.Triangulation(*(mesh_pos + u_obs[step]).T, cells)
    im = ax_c.tripcolor(trian, vals, cmap="Purples", shading="gouraud")
    ax_c.triplot(trian, color="#555555", lw=0.25, alpha=0.4)
    miss_node = miss_pred.reshape(-1, 2).any(axis=1)
    pos = (mesh_pos + u_obs[step])[free][miss_node]
    ax_c.plot(pos[:, 0], pos[:, 1], "x", color="#c0392b", ms=4.5, mew=1.2)
    ax_c.set_aspect("equal"); ax_c.axis("off")
    ax_c.set_title(rf"(c) $|\bar{{\boldsymbol{{\varepsilon}}}}|/\hat\sigma$, step {step} (x: observation outside predictive 95%)", fontsize=10.5)
    cax = make_axes_locatable(ax_c).append_axes("right", size="5%", pad=0.08)
    fig.colorbar(im, cax=cax).set_label(r"$|\bar{\boldsymbol{\varepsilon}}|/\hat\sigma$")

    # (d) coverage curves on the validation steps
    ax_d = fig.add_subplot(gs[2, 1])
    lv = LEVELS * 100
    ax_d.plot([0, 100], [0, 100], "k:", lw=1.0, label="Ideal")
    ax_d.plot(lv, cal_pred, "-", color="#1b4f72", lw=2.0, label=r"Observation-only: $0 \in$ interval of $\varepsilon_s + \hat\sigma z$")
    if cal_lat is not None:
        ax_d.plot(lv, cal_lat, "--", color="#b9770e", lw=1.8, label=r"Latent state (synthetic): $e \in$ interval of $\varepsilon_s$")
    ax_d.axvline(95, color="gray", lw=0.8, alpha=0.6)
    ax_d.set_xlim(0, 100); ax_d.set_ylim(0, 102); ax_d.set_box_aspect(1)
    ax_d.set_xlabel("Nominal coverage (%)"); ax_d.set_ylabel("Empirical coverage (%)")
    ax_d.grid(True, ls=":", alpha=0.5); ax_d.legend(loc="lower right", fontsize=7.8, framealpha=0.92)
    ax_d.set_title("(d) Coverage of the free-node DOFs, validation steps", fontsize=10.5)

    lat_val = cal_lat[i95] if cal_lat is not None else np.nan
    warn = (rf"   $\vert$   only {frac_stable[step] * 100:.0f}% of the samples stable at this step" if frac_stable[step] < 0.5 else "")
    excl = (f" (excluded, <50% samples stable: {', '.join(map(str, V_excl))})" if V_excl else "")
    banner = (rf"Step {step}: observation-only predictive coverage $= \mathbf{{{ec_pred_step:.1f}\%}}$"
              + (rf",  latent-state coverage $= {ec_lat_step:.1f}\%$" if e_true is not None else "") + warn
              + "\n" + rf"Validation steps{excl}: observation-only $= \mathbf{{{cal_pred[i95]:.1f}\%}}$"
              + (rf",  latent state $= {lat_val:.1f}\%$" if cal_lat is not None else "")
              + rf"   $\vert$   nominal 95%, {eps.shape[0]} posterior samples, $\hat\sigma = ({sig[0]:.2e}, {sig[1]:.2e})$")
    fig.text(0.5, -0.005, banner, ha="center", va="center", fontsize=9.0, linespacing=1.6,
             bbox=dict(boxstyle="round,pad=0.45", facecolor="#f5f5f5", edgecolor="#bbbbbb", lw=0.6))
    fig.suptitle(f"EIV correction at the free nodes and observation-only coverage (block, step {step})", fontsize=12.5, y=0.99)
    plt.subplots_adjust(left=0.06, right=0.97, top=0.95, bottom=0.07)
    out = out_dir / f"free_node_eps_step{step}_block.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)

    metrics = {"step": step, "frac_stable_step": float(frac_stable[step]), "val_steps_used": V, "val_steps_excluded": V_excl,
               "ec_obs_only_step": ec_pred_step, "ec_latent_step": ec_lat_step,
               "ec_obs_only_val": float(cal_pred[i95]), "ec_latent_val": float(lat_val),
               "calibration": {"nominal": lv.tolist(), "obs_only": cal_pred.tolist(),
                               "latent": cal_lat.tolist() if cal_lat is not None else None}}
    json.dump(metrics, open(out_dir / f"free_node_eps_step{step}_block.json", "w"), indent=2)
    print(f"[free-node] step {step}: obs-only EC {ec_pred_step:.1f}% latent EC {ec_lat_step:.1f}% | "
          f"val: obs-only {cal_pred[i95]:.1f}% latent {lat_val:.1f}% -> {out}")


if __name__ == "__main__":
    main()
