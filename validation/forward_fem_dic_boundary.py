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


def plot_fields_with_ends(X, cells, u_obs, results, models, ends, k, vs, steps, out_base, title):
    """Rows: models; columns: displacement magnitude on the deformed DIC mesh (measured outline dotted), nodal error
    |u_FEM - u_DIC|, parity (validation steps). The two prescribed end cuts are highlighted. Metrics of all models are
    stacked in one table under the figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri
    from matplotlib.lines import Line2D
    from plots.theme import apply_style, save_figure
    apply_style()
    nm = len(models)
    fig, axes = plt.subplots(nm, 3, figsize=(13.0, 3.9 * nm + 1.1), squeeze=False,
                             gridspec_kw=dict(width_ratios=[1, 1, 0.9]))
    end_col = {"end1": "#1565c0", "end2": "#e65100"}
    mags = [np.linalg.norm(results[q]["u"][k], axis=1) for q in models]
    errs = [np.linalg.norm(results[q]["u"][k] - u_obs[k], axis=1) for q in models]
    vmax, emax = max(m.max() for m in mags + [np.linalg.norm(u_obs[k], axis=1)]), max(e.max() for e in errs)
    obs_def = X + u_obs[k]
    rows = []

    def short(lab):
        return ("Distilled (mean)" if lab.startswith("Distilled model (mean") else
                lab.replace("Distilled model", "Distilled").replace(" (reference)", ""))
    lim = [min(u_obs[vs].min(), min(results[q]["u"][vs].min() for q in models)),
           max(u_obs[vs].max(), max(results[q]["u"][vs].max() for q in models))]
    for i, q in enumerate(models):
        u = results[q]["u"]
        P = X + u[k]
        tri = mtri.Triangulation(P[:, 0], P[:, 1], cells)
        a0, a1, a2 = axes[i]
        tp = a0.tripcolor(tri, mags[i], cmap="Blues", vmin=0, vmax=vmax, shading="gouraud")
        a0.triplot(mtri.Triangulation(obs_def[:, 0], obs_def[:, 1], cells), color="k", lw=0.15, alpha=0.35)
        te = a1.tripcolor(tri, errs[i], cmap="Oranges", vmin=0, vmax=emax, shading="gouraud")
        for ax in (a0, a1):
            for e, n in ends.items():
                ax.plot(P[n, 0], P[n, 1], "o", ms=2.6, color=end_col[e], zorder=5)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
        nm_q = short(results[q]["label"])
        a0.set_title(f"{nm_q}: |u| (step {int(steps[k])})", fontsize=10)
        a1.set_title(f"{nm_q}: error (step {int(steps[k])})", fontsize=10)
        fig.colorbar(tp, ax=a0, fraction=0.04, pad=0.01).set_label("mm")
        fig.colorbar(te, ax=a1, fraction=0.04, pad=0.01).set_label("mm")
        o, f_ = u_obs[vs].reshape(-1, 2), u[vs].reshape(-1, 2)
        a2.scatter(o[:, 0], f_[:, 0], s=2, color="#1565c0", alpha=0.25, lw=0, rasterized=True)
        a2.scatter(o[:, 1], f_[:, 1], s=2, color="#c62828", alpha=0.35, lw=0, rasterized=True)
        a2.plot(lim, lim, "k--", lw=0.9)
        a2.set_xlim(lim)
        a2.set_ylim(lim)
        a2.set_aspect("equal")
        if i == nm - 1:
            a2.set_xlabel("$u_{\\mathrm{DIC}}$ (mm)")
        else:
            a2.tick_params(labelbottom=False)
        a2.set_ylabel("$u_{\\mathrm{FEM}}$ (mm)")
        a2.set_title(f"{nm_q}: parity ({'all load steps' if len(vs) == len(steps) else 'selected steps'})", fontsize=10)
        a2.legend(handles=[Line2D([], [], marker="o", ls="", color="#1565c0", label="$u_x$"),
                           Line2D([], [], marker="o", ls="", color="#c62828", label="$u_y$")], fontsize=8, loc="upper left")
        a2.grid(alpha=0.25)
        r2 = [1 - np.sum((o[:, c] - f_[:, c]) ** 2) / np.sum((o[:, c] - o[:, c].mean()) ** 2) for c in (0, 1)]
        rm = np.sqrt(np.mean((o - f_) ** 2, axis=0))
        rows.append([nm_q,f"{r2[0]:.4f}", f"{r2[1]:.4f}", f"{rm[0]:.3f}", f"{rm[1]:.3f}",
                     f"{errs[i].mean():.2f}", f"{errs[i].max():.2f}"])
    tab = fig.add_axes([0.08, 0.005, 0.84, 0.035 * (nm + 1)])
    tab.axis("off")
    t = tab.table(cellText=rows, colLabels=["model", "$R^2_x$", "$R^2_y$", "RMSE$_x$ (mm)", "RMSE$_y$ (mm)",
                                            f"mean err, step {int(steps[k])}", f"max err, step {int(steps[k])}"],
                  loc="center", cellLoc="center", colWidths=[0.2] + [0.8 / 6] * 6)
    t.auto_set_font_size(False)
    t.set_fontsize(8.5)
    t.scale(1, 1.25)
    fig.legend(handles=[Line2D([], [], marker="o", ls="", color=end_col["end1"], label="end 1 (prescribed)"),
                        Line2D([], [], marker="o", ls="", color=end_col["end2"], label="end 2 (prescribed)"),
                        Line2D([], [], color="k", lw=0.6, alpha=0.5, label="measured deformed mesh")],
               loc="upper center", ncol=3, fontsize=9, frameon=False, bbox_to_anchor=(0.5, 0.965))
    fig.suptitle(title, y=0.995, fontsize=11)
    fig.tight_layout(rect=[0, 0.035 * (nm + 1) + 0.01, 1, 0.94])
    save_figure(fig, out_base + ".pdf")
    fig.set_dpi(170)
    fig.canvas.print_png(out_base + ".png")
    plt.close(fig)
    print(f"saved {out_base}.pdf/.png")


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
