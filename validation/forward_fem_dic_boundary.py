#!/usr/bin/env python3
"""
Forward FEM on the DIC data mesh (the region of interest) with the measured displacements prescribed on its two end
cuts (bcx = 1 and bcx = 2 nodes: both components, every load step); the specimen sides and the holes are traction
free. Incompressible plane stress, unit thickness. This removes the grip / clamp modelling of the full-specimen FEM
(validation/forward_fem_full_specimen.py): any remaining mismatch inside the ROI comes from the material model.

Models: the distilled model (--param_mode mean: one solve at the mean parameters; samples: --n_samples parameter
samples, mean field and 95% band) and a published reference model (default EUCLID).

Usage:
  python3 validation/forward_fem_dic_boundary.py --distilled_dir <seed>/distilled --out_dir <seed>/fem_validation/ttf_dic_bc \
      --dataset dataset/preprocessed/experiment/ttf_20_steps_ps_incomp.npz [--param_mode mean] [--plot_step 19]
Writes u_dicbc_<model>.npz (field, end-cut reaction) and displacement_analysis_dic_bc.pdf (+ reaction_dic_bc.pdf).
"""
import argparse
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "validation"))
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
jax.config.update("jax_enable_x64", True)
from forward_fem_full_specimen import (FullVectorMaterial, describe, distilled_full_vectors, grip_reaction,  # noqa: E402
                                       load_reference)


def node_set_fn(P):
    """One-argument location function: True at the given node positions."""
    P = jnp.asarray(P)
    return lambda p: jnp.min(jnp.sum((P - p) ** 2, axis=-1)) < 1e-10


def solve_dic_bc(mat, X, cells, ends, u_obs, label, max_substeps=32):
    from jax_fem.generate_mesh import Mesh
    from core.fem_engine import (BoundaryConditionConfig, DirichletBC, HyperElasticityProblem,
                                 make_incompressible_plane_stress_piola, solve_adaptive_fem)
    piola, _ = make_incompressible_plane_stress_piola(mat)
    bcs = []
    for name, nodes in ends.items():
        fn = node_set_fn(X[nodes])
        bcs += [DirichletBC(f"{name}_x", fn, 0, 0.0, is_prescribed=True), DirichletBC(f"{name}_y", fn, 1, 0.0, is_prescribed=True)]
    bc = BoundaryConditionConfig(mode="displacement", dirichlet_bcs=bcs)
    problem = HyperElasticityProblem(mesh=Mesh(X, cells), vec=2, dim=2, ele_type="TRI3",
                                     dirichlet_bc_info=bc.get_dirichlet_info(), location_fns=[], surface_maps=[],
                                     piola_func=lambda F: piola(F[:2, :2] if F.shape == (3, 3) else F))
    petsc = {"snes_type": "newtonls", "snes_linesearch_type": "bt", "snes_atol": 1e-10, "snes_rtol": 1e-10,
             "snes_stol": 1e-10, "snes_max_it": 50, "ksp_type": "preonly", "pc_type": "lu",
             "pc_factor_mat_solver_type": "mumps"}
    print(f"[{label}] solving {len(u_obs)} steps on the DIC mesh ({X.shape[0]} nodes) ...", flush=True)
    schedule = jnp.zeros((len(u_obs), 1))          # step count only: the values come from u_boundary_steps
    return np.asarray(solve_adaptive_fem(problem, bc, schedule, petsc, max_substeps=max_substeps, initial_substeps=2,
                                         u_boundary_steps=jnp.asarray(u_obs)))


def true_geometry_window(mesh_path, xq, dcells, u_obs, bcx):
    """The full specimen mesh cut to the DIC window (elements between the two end cuts, specimen frame) with the
    measured displacements interpolated onto its cut nodes. The DIC mesh stops short of the free edges and hole rims
    (by up to ~30 mm of width per section), so the window FEM on the DIC mesh is too compliant; this keeps the true
    geometry. Returns dict(X, cells, ends {end1, end2: node ids}, u_bc (steps, n, 2) specimen frame, A (2x2 frame map
    specimen -> DIC), X_dic, n_outside)."""
    import matplotlib.tri as mtri
    from scipy.spatial import cKDTree
    m = np.load(mesh_path, allow_pickle=True)
    X, cells = np.asarray(m["X"], float), np.asarray(m["tris"])
    M = np.linalg.lstsq(np.c_[X, np.ones(len(X))], np.asarray(m["X_dic"], float), rcond=None)[0]
    A, Ai = M[:2], np.linalg.inv(M[:2])
    xs, us = (xq - M[2]) @ Ai, u_obs @ Ai
    y1, y2 = xs[bcx == 1, 1].mean(), xs[bcx == 2, 1].mean()
    ylo, yhi = min(y1, y2), max(y1, y2)
    cen = X[cells].mean(1)[:, 1]
    sub = cells[(cen > ylo) & (cen < yhi)]
    used = np.unique(sub)
    mp = -np.ones(len(X), int)
    mp[used] = np.arange(len(used))
    Xs, Cs = X[used], mp[sub]
    h = float(np.median(np.linalg.norm(Xs[Cs[:, 0]] - Xs[Cs[:, 1]], axis=1)))
    lo_n, hi_n = np.where(Xs[:, 1] < ylo + 0.5 * h)[0], np.where(Xs[:, 1] > yhi - 0.5 * h)[0]
    ends = {"end1": lo_n if y1 < y2 else hi_n, "end2": hi_n if y1 < y2 else lo_n}
    cut = np.r_[lo_n, hi_n]
    tri = mtri.Triangulation(xs[:, 0], xs[:, 1], dcells)
    P = Xs[cut]
    out = tri.get_trifinder()(P[:, 0], P[:, 1]) < 0
    nn = cKDTree(xs).query(P)[1]
    ub = np.zeros((len(us), len(Xs), 2))
    for t in range(len(us)):
        for c in range(2):
            v = np.asarray(mtri.LinearTriInterpolator(tri, us[t][:, c])(P[:, 0], P[:, 1]).filled(np.nan))
            bad = out | np.isnan(v)
            v[bad] = us[t][nn[bad], c]
            ub[t, cut, c] = v
    print(f"true-geometry window: cuts at y = {ylo:.2f} / {yhi:.2f} (specimen frame), {len(Xs)} nodes, {len(Cs)} "
          f"elements; cut nodes {len(lo_n)} / {len(hi_n)} ({int(out.sum())} outside the DIC mesh: nearest DIC node)")
    return dict(X=Xs, cells=Cs, ends=ends, u_bc=ub, A=A, X_dic=Xs @ A + M[2])


REL_EPS = 1e-8      # relative error |u_FEM - u_DIC| / (|u_DIC| + REL_EPS): all nodes, REL_EPS only avoids 0 / 0


def plot_fields_with_ends(X, cells, u_obs, results, models, ends, k, vs, steps, out_base, title=None,
                          figsize=(210 / 25.4, None), max_maps=1):
    """Compact one-row figure (default: A4 width, height fitted to the panels): |u| of the first model (ours) on the
    deformed DIC mesh (measured deformed mesh on top), the nodal relative error eps_rel = 100 |u_FEM - u_DIC| / |u_DIC|
    of every model on the undeformed mesh (shared colour bar: where each model misses), and one parity plot of all
    models (validation steps, u_x and u_y). The two prescribed end cuts are highlighted; the metrics of all models in
    a table under the panels."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri
    from matplotlib.lines import Line2D
    from plots.theme import apply_style, save_figure
    apply_style()
    nm = len(models)
    end_col = {"end1": "#1565c0", "end2": "#e65100"}
    model_col = ["#009E73", "#757575", "#8e24aa", "#f9a825"]        # parity colours: ours first, then references
    # nodal relative error at the plotted step, all nodes
    mag_obs_k = np.linalg.norm(u_obs[k], axis=1)
    keep = np.ones_like(mag_obs_k, dtype=bool)
    errs = [100 * np.linalg.norm(results[q]["u"][k] - u_obs[k], axis=1) / (mag_obs_k + REL_EPS) for q in models]
    emax = max(np.nanmax(e) for e in errs)
    tri_ref = mtri.Triangulation(X[:, 0], X[:, 1], cells, mask=~np.all(keep[cells], axis=1))
    tri_all = mtri.Triangulation(X[:, 0], X[:, 1], cells)
    obs_def = X + u_obs[k]
    stp = "final step" if k == len(steps) - 1 else f"step {int(steps[k])}"     # name of the plotted load step

    def short(lab):     # every model is identified on the same data (UT + TTf): no data suffix in the labels
        return ("UGP + Distillation" if lab.startswith("Distilled model (mean") else
                lab.replace("Distilled model", "Distilled").replace(" (reference)", "").replace(" (UT+TTf)", ""))
    names = [short(results[q]["label"]) for q in models]
    n_maps = min(nm, max_maps)          # error maps / parity: the first n_maps models; the table: all models
    lim = [min(u_obs[vs].min(), min(results[q]["u"][vs].min() for q in models)),
           max(u_obs[vs].max(), max(results[q]["u"][vs].max() for q in models))]
    w_ref = max(np.ptp(X[:, 0]) / np.ptp(X[:, 1]), 0.35)
    P0 = X + results[models[0]]["u"][k]
    w_def = np.ptp(P0[:, 0]) / np.ptp(P0[:, 1])
    # explicit layout in inches: [|u| ours][cbar+label][error map(s)][cbar+label][parity ylabel][parity], the panel
    # height chosen so the row fills the width; colour bars sit right next to their panel
    W = figsize[0]
    m_l, m_r, m_t, m_b = 0.08, 0.06, 0.42, 0.36           # margins (legend + titles on top, parity x label below)
    cb_w, cb_lab, gap, y_lab = 0.09, 0.50, 0.10, 0.45      # colour bar, its ticks + label, panel gap, parity y label
    w_panels = w_def + n_maps * w_ref + 1.0                # widths in units of the panel height
    fixed = m_l + m_r + 2 * (gap + cb_w + cb_lab) + (n_maps - 1) * gap + y_lab
    h = (W - fixed) / w_panels
    H = figsize[1] if figsize[1] is not None else h + m_t + m_b
    fig = plt.figure(figsize=(W, H))
    y0, hh = m_b / H, h / H

    def put(x, w):
        return fig.add_axes([x / W, y0, w / W, hh]), x + w
    x = m_l
    a0, x = put(x, h * w_def)
    cax_u, x = put(x + gap, cb_w)
    x += cb_lab
    err_axes = []
    for _ in range(n_maps):
        a_e, x = put(x, h * w_ref)
        err_axes.append(a_e)
        x += gap
    cax_e, x = put(x, cb_w)
    x += cb_lab + y_lab
    a2, x = put(x, h)
    fs, fs_t = 7.0, 7.3

    def field_axes(ax, Q):
        for e, n in ends.items():
            ax.plot(Q[n, 0], Q[n, 1], "o", ms=1.0, color=end_col[e], zorder=5)
        ax.set_aspect("equal")
        ax.set_anchor("C")
        ax.axis("off")

    def cbar(cax, mappable, lab):
        cb = fig.colorbar(mappable, cax=cax)
        cb.set_label(lab, fontsize=fs, labelpad=2)
        cb.ax.tick_params(labelsize=6, length=2, pad=1)
        cb.outline.set_linewidth(0.5)
    # |u| of ours on the deformed mesh
    tp = a0.tripcolor(mtri.Triangulation(P0[:, 0], P0[:, 1], cells), np.linalg.norm(results[models[0]]["u"][k], axis=1),
                      cmap="Blues", vmin=0, vmax=max(mag_obs_k.max(), np.linalg.norm(results[models[0]]["u"][k], axis=1).max()),
                      shading="gouraud", rasterized=True)
    a0.triplot(mtri.Triangulation(obs_def[:, 0], obs_def[:, 1], cells), color="k", lw=0.06, alpha=0.35)
    field_axes(a0, P0)
    a0.set_title((f"{names[0]}: " if n_maps > 1 else "") + rf"$|\mathbf{{u}}|$, deformed ({stp})", fontsize=fs_t, pad=3)
    cbar(cax_u, tp, r"$|\mathbf{u}|$ (mm)")
    # relative error of every model on the undeformed mesh
    for i, q in enumerate(models[:n_maps]):
        a = err_axes[i]
        a.tripcolor(tri_all, np.zeros(len(X)), cmap="Greys", vmin=0, vmax=4, shading="flat", zorder=0, rasterized=True)
        te = a.tripcolor(tri_ref, np.nan_to_num(errs[i]), cmap="Oranges", vmin=0, vmax=emax, shading="gouraud",
                         rasterized=True)
        field_axes(a, X)
        a.set_title((f"{names[i]}:\n" if n_maps > 1 else "") + r"$\epsilon_{\mathrm{rel}}$, undeformed", fontsize=fs_t, pad=3)
    cbar(cax_e, te, r"$\epsilon_{\mathrm{rel}}$ (%)")
    # parity of all models (references underneath, ours on top)
    o = u_obs[vs].reshape(-1)
    if n_maps == 1:        # one model: colour by component instead of by model
        ov, fv = u_obs[vs].reshape(-1, 2), results[models[0]]["u"][vs].reshape(-1, 2)
        a2.scatter(ov[:, 0], fv[:, 0], s=0.6, color="#1565c0", alpha=0.25, lw=0, rasterized=True)
        a2.scatter(ov[:, 1], fv[:, 1], s=0.6, color="#c62828", alpha=0.35, lw=0, rasterized=True)
        a2.legend(handles=[Line2D([], [], marker="o", ls="", ms=3, color="#1565c0", label="$u_x$"),
                           Line2D([], [], marker="o", ls="", ms=3, color="#c62828", label="$u_y$")], fontsize=6,
                  loc="upper left", frameon=False, handletextpad=0.2, borderaxespad=0.3)
    for i in (list(range(1, n_maps)) + [0] if n_maps > 1 else []):
        f_ = results[models[i]]["u"][vs].reshape(-1)
        a2.scatter(o, f_, s=2.2 if i else 0.5, color=model_col[i % len(model_col)], alpha=0.35 if i else 0.3, lw=0,
                   rasterized=True)                       # references larger underneath: their spread shows as a rim
    a2.plot(lim, lim, "k--", lw=0.6)
    a2.set_xlim(lim)
    a2.set_ylim(lim)
    a2.set_box_aspect(1)
    a2.grid(False)
    a2.tick_params(labelsize=6, length=2, pad=1.5)
    a2.set_xlabel(r"$u_{\mathrm{DIC}}$ (mm)", fontsize=fs, labelpad=1)
    a2.set_ylabel(r"$u_{\mathrm{FEM}}$ (mm)", fontsize=fs, labelpad=1)
    a2.set_title("parity ($u_x$, $u_y$; " + ("all load steps)" if len(vs) == len(steps) else "selected steps)"),
                 fontsize=fs_t, pad=3)
    if n_maps > 1:
        a2.legend(handles=[Line2D([], [], marker="o", ls="", ms=3, color=model_col[i % len(model_col)], label=names[i])
                           for i in range(n_maps)], fontsize=5.8, loc="upper left", frameon=False, handletextpad=0.2,
                  borderaxespad=0.3)
    # metrics of all models -> LaTeX table (booktabs) next to the figure; best value per column in bold
    rows = []
    for i, q in enumerate(models):
        ov, fv = u_obs[vs].reshape(-1, 2), results[q]["u"][vs].reshape(-1, 2)
        mo, mf = np.linalg.norm(ov, axis=1), np.linalg.norm(fv, axis=1)          # displacement magnitudes
        r2_mag = 1 - np.sum((mo - mf) ** 2) / np.sum((mo - mo.mean()) ** 2)
        # relative error of the displacement vector over all nodes and load steps (|u_DIC| > 0 everywhere, unlike
        # the components: u_y changes sign across the specimen)
        rel_all = 100 * np.mean(np.linalg.norm(fv - ov, axis=1) / (np.linalg.norm(ov, axis=1) + REL_EPS))
        rows.append([r2_mag, float(np.nanmean(errs[i]))])
    rows = np.array(rows)
    best = [rows[:, 0].argmax(), rows[:, 1].argmin()]
    fmt = ["{:.4f}", "{:.2f}"]

    def cell(r, c):
        v = fmt[c].format(rows[r, c])
        return rf"\textbf{{{v}}}" if best[c] == r else v
    body = [f"{names[r]} & " + " & ".join(cell(r, c) for c in range(2)) + r" \\" for r in range(nm)]
    tex = "\n".join([
        r"\begin{table}[t]", r"\centering", r"\small",
        r"\begin{tabular}{lcc}", r"\toprule",
        r"Model & $R^2_{\|\mathbf{u}\|}$ & $\bar{\epsilon}_{\mathrm{rel}}$ (\%) \\", r"\midrule", *body, r"\bottomrule", r"\end{tabular}",
        r"\caption{Displacement prediction on the TTf specimen with the measured DIC displacements prescribed on the "
        r"two end cuts. $R^2_{\|\mathbf{u}\|}$: coefficient of determination of the displacement magnitude over all "
        r"nodes and load steps; $\bar{\epsilon}_{\mathrm{rel}}$: mean over the nodes of the relative error of the "
        r"displacement vector $|\mathbf{u}_{\mathrm{FEM}} - \mathbf{u}_{\mathrm{DIC}}|\,/\,(|\mathbf{u}_{\mathrm{DIC}}| "
        r"+ \delta)$, $\delta = 10^{-8}$, at the " + stp + r". Best value in bold.}",
        r"\label{tab:dic_bc_metrics}", r"\end{table}"])
    with open(out_base.replace("displacement_analysis", "displacement_metrics") + ".tex", "w") as f:
        f.write(tex + "\n")
    print(tex)
    fig.legend(handles=[Line2D([], [], marker="o", ls="", ms=3, color=end_col["end1"], label="end 1 (prescribed)"),
                        Line2D([], [], marker="o", ls="", ms=3, color=end_col["end2"], label="end 2 (prescribed)"),
                        Line2D([], [], color="k", lw=0.6, alpha=0.5, label="measured deformed mesh")],
               loc="upper center", ncol=3, fontsize=6.3, frameon=False, bbox_to_anchor=(0.5, 1.0),
               handletextpad=0.3, columnspacing=1.5)
    if title:
        print(f"[plot] {title}")
    save_figure(fig, out_base + ".pdf")
    fig.set_dpi(250)
    fig.canvas.print_png(out_base + ".png")
    plt.close(fig)

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--distilled_dir", default=None, help="distilled model folder (omit: reference model only)")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--dataset", default=os.path.join(ROOT, "dataset", "preprocessed", "experiment", "ttf_20_steps_ps_incomp.npz"))
    p.add_argument("--reference", default="euclid")
    p.add_argument("--extra_references", nargs="*", default=[],
                   help="more reference models (keys of plots/plot_reference_energy.REFERENCES), one row / curve each")
    p.add_argument("--param_mode", choices=["mean", "samples"], default="mean")
    p.add_argument("--n_samples", type=int, default=32)
    p.add_argument("--sample_seed", type=int, default=0)
    p.add_argument("--plot_step", type=int, default=-1)
    p.add_argument("--val_steps", type=int, nargs="*", default=None)
    p.add_argument("--plot_only", action="store_true", help="reuse saved u_dicbc_*.npz")
    p.add_argument("--full_mesh", default=None,
                   help="specimen mesh with X_dic (e.g. mesh/TTf_full_mesh.npz): solve on the true geometry of the "
                        "window (the full mesh cut at the DIC end cuts) instead of the DIC mesh, which stops short of "
                        "the free edges and hole rims")
    p.add_argument("--thickness", type=float, default=None,
                   help="specimen thickness (mm): the force plot shows the total force in N (= force per thickness x "
                        "thickness); default: force per unit thickness (N/mm)")
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    ds = np.load(args.dataset, allow_pickle=True)
    X, cells = np.asarray(ds["mesh_pos"])[:, :2].astype(float), np.asarray(ds["cells"])
    u_obs, steps = np.asarray(ds["u_exp"], float), np.asarray(ds["steps"])
    R_meas = np.asarray(ds["reaction_forces"])
    bcx = np.asarray(ds["bcx"]).ravel()
    ends = {"end1": np.where(bcx == 1)[0], "end2": np.where(bcx == 2)[0]}
    load_dof = int(np.argmax(np.abs(R_meas[-1])))
    end2_moves = abs(u_obs[-1, ends["end2"], load_dof].mean()) > abs(u_obs[-1, ends["end1"], load_dof].mean())
    pulled = "end2" if end2_moves else "end1"
    print(f"prescribed ends: {len(ends['end1'])} / {len(ends['end2'])} nodes (bcx 1 / 2); reaction from {pulled} "
          f"(the end that moves more), component {load_dof}")

    if args.full_mesh:      # true geometry: solve in the specimen frame, map back to the DIC nodes for the plots
        from forward_fem_full_specimen import to_dic_nodes
        geo = true_geometry_window(args.full_mesh, X, cells, u_obs, bcx)
        Xf, Cf, ends_f, u_bc = geo["X"], geo["cells"], geo["ends"], geo["u_bc"]
    else:
        Xf, Cf, ends_f, u_bc = X, cells, ends, u_obs

    dist_key = "distilled_mean" if args.param_mode == "mean" else "distilled_samples"
    models = ([dist_key] if args.distilled_dir else []) + list(dict.fromkeys([args.reference] + args.extra_references))
    results = {}
    for key in models:
        f = os.path.join(args.out_dir, f"u_dicbc_{key}.npz")
        if args.plot_only and os.path.exists(f):
            d = np.load(f, allow_pickle=True)
            results[key] = dict(u=d["u"], label=str(d["label"]), R=d["reaction"],
                                lo=d["u_lo"] if "u_lo" in d.files else None, hi=d["u_hi"] if "u_hi" in d.files else None)
            continue
        if not key.startswith("distilled"):
            mats, thetas = [load_reference(key)], [None]
            label = f"{mats[0].label} (reference)"
        else:
            P = distilled_full_vectors(args.distilled_dir)
            if key == "distilled_mean":
                thetas, label = [P.mean(0)], "Distilled model (mean parameters)"
                print(f"distilled mean parameters: {describe(thetas[0])}")
            else:
                idx = np.random.default_rng(args.sample_seed).choice(len(P), min(args.n_samples, len(P)), replace=False)
                thetas, label = list(P[idx]), "Distilled model (parameter samples, mean field)"
            mats = [FullVectorMaterial(t) for t in thetas]
        U, R = [], []
        for i, mat in enumerate(mats):
            tag = label if len(mats) == 1 else f"{label}: sample {i + 1}/{len(mats)}"
            try:
                u = solve_dic_bc(mat, Xf, Cf, ends_f, u_bc, tag)
                if not np.all(np.isfinite(u)):
                    raise RuntimeError("non-finite displacements")
            except Exception as exc:
                print(f"[{tag}] FEM failed: {exc}")
                continue
            r = grip_reaction(mat, Xf, Cf, u, ends_f[pulled])
            if args.full_mesh:      # specimen frame -> DIC frame (displacements at the DIC nodes, force vector)
                u = to_dic_nodes(u, geo["X_dic"], Cf, X, Amat=geo["A"])[0]
                r = r @ geo["A"]
            U.append(u)
            R.append(r)
        if not U:
            raise SystemExit(f"[{label}] no successful FEM solve")
        U = np.stack(U)
        extra, lo, hi = {}, None, None
        if len(U) > 1:
            lo, hi = np.percentile(U, 2.5, axis=0), np.percentile(U, 97.5, axis=0)
            extra = dict(u_lo=lo, u_hi=hi, reaction_samples=np.array(R))
            label = label.replace("parameter samples", f"{len(U)} parameter samples")
        u_m, R_m = U.mean(0), np.mean(R, 0)
        np.savez_compressed(f, u=u_m, reaction=R_m, steps=steps, label=label, **extra)
        print(f"[{label}] saved {f}; end force at the last step {np.round(R_m[-1], 3)}, measured {np.round(R_meas[-1], 3)}")
        results[key] = dict(u=u_m, label=label, R=R_m, lo=lo, hi=hi)

    # --- displacement figure (DIC frame), distilled first, reference as an extra row
    from plots.uq_verification_disp import plot_disp_field
    vs = np.arange(len(steps)) if args.val_steps is None else np.asarray(args.val_steps)
    k = args.plot_step if args.plot_step >= 0 else len(steps) + args.plot_step
    main_key = models[0]
    mr = results[main_key]
    lo_v = None if mr["lo"] is None else mr["lo"][vs].reshape(-1, 2)
    hi_v = None if mr["hi"] is None else mr["hi"][vs].reshape(-1, 2)
    extra = [dict(label=results[q]["label"], u_step=results[q]["u"][k], u_val_flat=results[q]["u"][vs].reshape(-1, 2))
             for q in models[1:]]
    plot_fields_with_ends(X, cells, u_obs, results, models, ends, k, vs, steps,
                          os.path.join(args.out_dir, "displacement_analysis_dic_bc"),
                          "True window geometry, measured end displacements (fields shown on the DIC mesh)"
                          if args.full_mesh else "DIC mesh, measured end displacements")

    # --- reaction force: end force vs the measured machine force
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from plots.theme import apply_style, save_figure
    apply_style()
    t = args.thickness or 1.0
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    ax.plot(steps, t * np.abs(R_meas[:, load_dof]), "ko", ms=4, label="measured")
    for q, c in zip(models, ["#1565c0", "#6a1b9a", "#2e7d32"]):
        ax.plot(steps, t * np.abs(results[q]["R"][:, load_dof]), "-", color=c, lw=1.6, label=results[q]["label"])
    ax.set_xlabel("DIC step")
    ax.set_ylabel("reaction force (N)" if args.thickness else "reaction force per thickness (N/mm)")
    ax.set_title(("Force at the moving end, true window geometry" if args.full_mesh else "Force at the moving end, DIC-mesh FEM")
                 + (f" (thickness {t:g} mm)" if args.thickness else ""), fontsize=10.5)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    save_figure(fig, os.path.join(args.out_dir, "reaction_dic_bc.pdf"))
    plt.close(fig)
    for q in models:
        r = np.abs(results[q]["R"][:, load_dof]) / np.maximum(np.abs(R_meas[:, load_dof]), 1e-12)
        print(f"{results[q]['label']}: force / measured per step: {np.round(r, 3)}")


if __name__ == "__main__":
    main()
