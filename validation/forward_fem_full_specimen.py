#!/usr/bin/env python3
"""
Forward FEM of the whole TTc specimen (mesh/TTc_full_mesh.npz: 50 x 205 mm with holes, grips at y = 0 and y = 205)
under displacement control, compared with the DIC field of the region of interest.

Loading (from the test): the top grip (y = 205) is held, the bottom grip (y = 0) is pulled down; the grip displacement
rises linearly with the DIC step number to 100 mm at step 6000 (u_grip = 100 mm x step / 6000). Both grips are clamped
in x. Incompressible plane stress (F33 = 1 / det F_2D), unit thickness.

Models: the published EUCLID model (or any key of plots/plot_reference_energy.REFERENCES) and the distilled model
(GMR and/or Ogden library): --param_mode mean (default) solves once with the mean of the parameter samples
(u_full_distilled_mean.npz); --param_mode samples solves --n_samples parameter samples and reports the mean field with
the 2.5-97.5% band (u_full_distilled_samples.npz). The FEM displacements are mapped to the DIC frame (x_dic = tx - y, y_dic = x + ty,
so u_dic = (-u_y, u_x)) and interpolated at the DIC nodes.

Usage:
  python3 validation/forward_fem_full_specimen.py --distilled_dir <seed>/distilled --out_dir <seed>/fem_validation/ttc_full
Writes u_full_<model>.npz (full-mesh field, field at the DIC nodes, grip reaction) and displacement_analysis_full_mesh.pdf
(DIC field vs the distilled-mean model, plus a row for the reference model).
"""
import argparse
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
jax.config.update("jax_enable_x64", True)


def load_reference(key):
    import importlib.util
    spec = importlib.util.spec_from_file_location("plot_reference_energy", os.path.join(ROOT, "plots", "plot_reference_energy.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.ReferenceMaterial(key)


def distilled_full_vectors(distilled_dir):
    """Distilled parameter samples as full library vectors (S, 13 | N_FULL): GMR and/or Ogden terms."""
    from core.material_library import distilled_library, full_param_vectors
    dev = np.load(os.path.join(distilled_dir, "dev_flow_samples.npy"))
    f_vol = os.path.join(distilled_dir, "vol_flow_samples.npy")
    vol = np.load(f_vol) if os.path.exists(f_vol) else np.zeros((dev.shape[0], 3))
    return full_param_vectors(dev, vol[:, :3], distilled_library(distilled_dir))


class FullVectorMaterial:
    """Energy of one full parameter vector: GMR polynomial (+ log) terms and Ogden terms. Under incompressible plane
    stress J = 1, so the volumetric part drops out."""

    def __init__(self, p):
        from core.material_models import get_material
        p = np.asarray(p, float)
        self.p = p
        self._gmr = get_material("gmr", dev_params=list(map(float, p[:10])), vol_params=list(map(float, p[10:13])),
                                 jit_P=False)
        og = p[19:] if p.shape[0] > 19 else np.zeros(0)
        self._mus, self._alphas = jnp.asarray(og[0::2]), jnp.asarray(og[1::2])
        self._has_ogden = bool(np.any(og[0::2] != 0.0))

    def psi(self, F):
        w = self._gmr.psi(F)
        if self._has_ogden:
            from core.ogden import ogden_dev_energy_jax
            w = w + ogden_dev_energy_jax(F, self._mus, self._alphas)
        return w


def describe(p):
    from core.material_library import FULL_NAMES
    return ", ".join(f"{n} {v:.4g}" for n, v in zip(FULL_NAMES, p) if abs(v) > 1e-12)


def clamp_line_nodes(X, cells, y_c):
    """Nodes on (closest to) the horizontal line y = y_c: within half the median edge length of it."""
    e = np.vstack([cells[:, [0, 1]], cells[:, [1, 2]], cells[:, [2, 0]]])
    h = float(np.median(np.linalg.norm(X[e[:, 0]] - X[e[:, 1]], axis=1)))
    tol = 0.5 * h        # widen the band until the clamped nodes cover the width (no x gap above 1.5 h)
    while True:
        nodes = np.where(np.abs(X[:, 1] - y_c) <= tol)[0]
        xs = np.sort(np.r_[X[nodes, 0], X[:, 0].min(), X[:, 0].max()])
        if np.diff(xs).max() <= 1.5 * h or tol > 3 * h:
            return nodes, tol
        tol += 0.1 * h


def solve(mat, X, cells, at_held, at_pulled, load_dof, load_sign, grip, label, max_substeps=32):
    """Held grip: both components 0; pulled grip: the other component 0, the load component = load_sign x grip."""
    from jax_fem.generate_mesh import Mesh
    from core.fem_engine import (BoundaryConditionConfig, DirichletBC, HyperElasticityProblem,
                                 make_incompressible_plane_stress_piola, solve_adaptive_fem)
    piola, _ = make_incompressible_plane_stress_piola(mat)
    other = 1 - load_dof
    bc = BoundaryConditionConfig(mode="displacement", dirichlet_bcs=[
        DirichletBC("held_x", at_held, 0, 0.0), DirichletBC("held_y", at_held, 1, 0.0),
        DirichletBC("pulled_other", at_pulled, other, 0.0),
        DirichletBC("pulled_load", at_pulled, load_dof, 0.0, is_prescribed=True, schedule_index=0)])
    problem = HyperElasticityProblem(mesh=Mesh(X, cells), vec=2, dim=2, ele_type="TRI3",
                                     dirichlet_bc_info=bc.get_dirichlet_info(), location_fns=[], surface_maps=[],
                                     piola_func=lambda F: piola(F[:2, :2] if F.shape == (3, 3) else F))
    petsc = {"snes_type": "newtonls", "snes_linesearch_type": "bt", "snes_atol": 1e-10, "snes_rtol": 1e-10,
             "snes_stol": 1e-10, "snes_max_it": 50, "ksp_type": "preonly", "pc_type": "lu",
             "pc_factor_mat_solver_type": "mumps"}
    schedule = jnp.asarray(load_sign * grip[:, None])
    print(f"[{label}] solving {len(grip)} steps on the full mesh ({X.shape[0]} nodes) ...", flush=True)
    return np.asarray(solve_adaptive_fem(problem, bc, schedule, petsc, max_substeps=max_substeps, initial_substeps=2))


def grip_reaction(mat, X, cells, u, nodes):
    """Sum of the internal forces at the given grip nodes (per unit thickness), per step: (steps, 2)."""
    from core.material_models import IncompressiblePlaneStressView
    psi = IncompressiblePlaneStressView(mat)._reduced_psi
    Xc = X[cells]                                                    # (C, 3, 2)
    J = np.stack([Xc[:, 1] - Xc[:, 0], Xc[:, 2] - Xc[:, 0]], axis=-1)   # columns: edge vectors
    Jinv = np.linalg.inv(J)
    dNdxi = np.array([[-1.0, -1.0], [1.0, 0.0], [0.0, 1.0]])
    dNdX = jnp.asarray(np.einsum("ak,ckj->caj", dNdxi, Jinv))          # (C, 3, 2)
    dA = jnp.asarray(0.5 * np.abs(np.linalg.det(J)))
    P_fn = jax.vmap(jax.grad(lambda f2: psi(jnp.zeros((3, 3)).at[:2, :2].set(f2).at[2, 2].set(1.0))))
    out = []
    for u_t in u:
        F2 = jnp.eye(2) + jnp.einsum("cai,caj->cij", jnp.asarray(u_t[cells]), dNdX)
        P = P_fn(F2)
        f_cell = jnp.einsum("cij,caj->cai", P, dNdX) * dA[:, None, None]
        f = jnp.zeros((X.shape[0], 2)).at[jnp.asarray(cells)].add(f_cell)
        out.append(np.asarray(f[nodes].sum(0)))
    return np.array(out)


def to_dic_nodes(u, X_dic, cells, xq, Amat=None):
    """Mesh-frame displacements (steps, n, 2) -> DIC frame (u_dic = u @ Amat, the linear part of X -> X_dic; None: the
    mesh is already in the DIC frame), interpolated at the DIC nodes xq (m, 2)."""
    to_d = (lambda v: v @ Amat) if Amat is not None else (lambda v: v)
    import matplotlib.tri as mtri
    tri = mtri.Triangulation(X_dic[:, 0], X_dic[:, 1], cells)
    finder = tri.get_trifinder()
    out = np.zeros((u.shape[0], len(xq), 2))
    outside = finder(xq[:, 0], xq[:, 1]) < 0
    for t in range(u.shape[0]):
        ud = to_d(u[t])                                              # DIC frame components
        for c in range(2):
            v = mtri.LinearTriInterpolator(tri, ud[:, c])(xq[:, 0], xq[:, 1])
            out[t, :, c] = np.ma.filled(v, np.nan)
    if outside.any():                                                # DIC nodes just outside the mesh: nearest node
        from scipy.spatial import cKDTree
        idx = cKDTree(X_dic).query(xq[outside])[1]
        for t in range(u.shape[0]):
            out[t, outside] = to_d(u[t, idx])
    return out, int(outside.sum())


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--distilled_dir", default=None, help="distilled model folder (omit: reference model only)")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--mesh", default=os.path.join(ROOT, "mesh", "TTc_full_mesh.npz"))
    p.add_argument("--dataset", default=os.path.join(ROOT, "dataset", "preprocessed", "experiment", "ttc_20_steps_ps_incomp.npz"))
    p.add_argument("--grip_max", type=float, default=100.0, help="grip displacement at --step_max (mm)")
    p.add_argument("--step_max", type=float, default=6000.0)
    p.add_argument("--reference", default="euclid")
    p.add_argument("--plot_step", type=int, default=16, help="load step index of the field panels")
    p.add_argument("--val_steps", type=int, nargs="*", default=None, help="steps of the parity panel (default: all)")
    p.add_argument("--plot_only", action="store_true", help="reuse saved u_full_*.npz")
    p.add_argument("--param_mode", choices=["mean", "samples"], default="mean",
                   help="distilled model: mean of the parameter samples (one solve), or n_samples parameter samples "
                        "(mean field and 95%% band)")
    p.add_argument("--n_samples", type=int, default=32, help="--param_mode samples: number of parameter samples")
    p.add_argument("--sample_seed", type=int, default=0)
    p.add_argument("--clamp_bottom_y", type=float, default=None,
                   help="pulled grip: clamp the nodes on the horizontal line closest to this y instead of the bottom edge")
    p.add_argument("--clamp_top_y", type=float, default=None,
                   help="held grip: clamp the nodes on the horizontal line closest to this y instead of the top edge")
    p.add_argument("--clamp_distance", type=float, default=None,
                   help="TTe / TTf layouts: clamp everything within this distance of each end (instead of the mesh's tab regions)")
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    m = np.load(args.mesh, allow_pickle=True)
    X, cells = np.asarray(m["X"], float), np.asarray(m["tris"])
    ds = np.load(args.dataset, allow_pickle=True)
    steps = np.asarray(ds["steps"], float)
    grip = args.grip_max * steps / args.step_max
    xq, dic_cells, u_obs = np.asarray(ds["mesh_pos"])[:, :2], np.asarray(ds["cells"]), np.asarray(ds["u_exp"])
    R_meas = np.asarray(ds["reaction_forces"])
    if "X_dic" in m.files:
        # specimen frame (loading along y, grips at y = 0 and y = L) with the DIC frame via X_dic (affine):
        # TTc (end edges, or clamp lines --clamp_bottom_y / --clamp_top_y) and TTf (tab regions tab_bottom / tab_top,
        # or --clamp_distance from each end). Which grip is pulled, and in which direction, comes from the DIC data.
        Xd = np.asarray(m["X_dic"], float)
        M = np.linalg.lstsq(np.c_[X, np.ones(len(X))], Xd, rcond=None)[0]
        Amat, load_dof = M[:2], 1
        nb_edge, nt_edge = np.asarray(m["nodes_bottom"]), np.asarray(m["nodes_top"])
        y_lo, y_hi = float(X[:, 1].min()), float(X[:, 1].max())
        names = [str(n) for n in np.asarray(m["region_names"])] if "region_names" in m.files else []

        def near_line(y_c, tol):     # one-argument location functions (jax-fem checks the signature)
            return lambda p: jnp.abs(p[1] - y_c) <= tol + 1e-9

        def below(y_c):
            return lambda p: p[1] <= y_c + 1e-6

        def above(y_c):
            return lambda p: p[1] >= y_c - 1e-6
        if args.clamp_bottom_y is not None or args.clamp_top_y is not None:          # clamp lines
            nb, tol_b = clamp_line_nodes(X, cells, args.clamp_bottom_y) if args.clamp_bottom_y is not None else (nb_edge, 1e-6)
            nt, tol_t = clamp_line_nodes(X, cells, args.clamp_top_y) if args.clamp_top_y is not None else (nt_edge, 1e-6)
            at_bottom, at_top = near_line(float(X[nb, 1].mean()), tol_b), near_line(float(X[nt, 1].mean()), tol_t)
            how = f"clamp lines at y = {X[nb, 1].mean():.2f} / {X[nt, 1].mean():.2f} ({len(nb)} / {len(nt)} nodes)"
        elif args.clamp_distance is not None or ("tab_bottom" in names and "tab_top" in names):   # clamped regions
            if args.clamp_distance is not None:
                yb, yt = y_lo + args.clamp_distance, y_hi - args.clamp_distance
            else:
                reg = np.asarray(m["region"])
                yb = float(X[np.unique(cells[reg == names.index("tab_bottom")]), 1].max())
                yt = float(X[np.unique(cells[reg == names.index("tab_top")]), 1].min())
            nb, nt = np.where(X[:, 1] <= yb + 1e-6)[0], np.where(X[:, 1] >= yt - 1e-6)[0]
            at_bottom, at_top = below(yb), above(yt)
            how = (f"clamped regions y <= {yb:.2f} and y >= {yt:.2f} ({len(nb)} / {len(nt)} nodes; "
                   f"{yb - y_lo:.1f} / {y_hi - yt:.1f} mm from the ends)")
        else:                                                                         # end edges
            nb, nt = nb_edge, nt_edge
            at_bottom, at_top = near_line(y_lo, 1e-6), near_line(y_hi, 1e-6)
            how = "end edges"
        # pulled grip: the DIC window end nearer to it moves more along the loading axis (specimen +y in the DIC frame)
        e = np.array([0.0, 1.0]) @ Amat
        sq = xq @ e
        s_top_grip = float(Xd[nt_edge].mean(0) @ e) > float(Xd[nb_edge].mean(0) @ e)
        end_hi, end_lo = sq > sq.max() - 1.0, sq < sq.min() + 1.0
        u_e = u_obs[-1] @ e
        top_end, bot_end = (end_hi, end_lo) if s_top_grip else (end_lo, end_hi)
        top_pulled = abs(u_e[top_end].mean()) > abs(u_e[bot_end].mean())
        load_sign = float(np.sign(u_e[top_end].mean() if top_pulled else u_e[bot_end].mean()))
        at_pulled, at_held = (at_top, at_bottom) if top_pulled else (at_bottom, at_top)
        held_nodes = nb if top_pulled else nt
        print(f"grips: {how}; {'top' if top_pulled else 'bottom'} grip pulled along {'+' if load_sign > 0 else '-'}y, "
              f"the other held")
    else:
        # TTe layout: the mesh is in the DIC frame (loading along x) with the gripped tabs as node sets
        # (clamp_bottom / clamp_top regions); the tab on the side the DIC field moves towards is pulled
        Xd, Amat, load_dof = X, None, 0
        tabs = {k: np.asarray(m[k]) for k in ("clamp_bottom", "clamp_top")}
        side = {k: float(X[n, 0].mean()) for k, n in tabs.items()}
        right = max(side, key=side.get)
        left = min(side, key=side.get)
        u_last = u_obs[-1, :, 0]
        moves_right = abs(u_last[xq[:, 0] > xq[:, 0].max() - 1].mean()) > abs(u_last[xq[:, 0] < xq[:, 0].min() + 1].mean())
        pulled, held = (right, left) if moves_right else (left, right)
        load_sign = float(np.sign(u_last[xq[:, 0] > xq[:, 0].max() - 1].mean() if moves_right
                                  else u_last[xq[:, 0] < xq[:, 0].min() + 1].mean()))
        x_r_min, x_l_max = float(X[tabs[right], 0].min()), float(X[tabs[left], 0].max())
        if args.clamp_distance is not None:     # clamp regions of a given length from the specimen ends
            x_r_min, x_l_max = float(X[:, 0].max()) - args.clamp_distance, float(X[:, 0].min()) + args.clamp_distance
            tabs = {right: np.where(X[:, 0] >= x_r_min - 1e-6)[0], left: np.where(X[:, 0] <= x_l_max + 1e-6)[0]}
            print(f"clamp regions {args.clamp_distance} mm from the ends: x <= {x_l_max:.3f} and x >= {x_r_min:.3f}")
        sel_right = lambda p: p[0] >= x_r_min - 1e-6
        sel_left = lambda p: p[0] <= x_l_max + 1e-6
        at_pulled, at_held = (sel_right, sel_left) if moves_right else (sel_left, sel_right)
        for nm, fn, n in ((pulled, at_pulled, tabs[pulled]), (held, at_held, tabs[held])):
            sel = int(np.sum(np.asarray(jax.vmap(fn)(jnp.asarray(X)))))
            print(f"{'pulled' if nm == pulled else 'held'} tab '{nm}': {sel} nodes selected ({len(n)} in the node set), "
                  f"x {X[n, 0].min():.2f}..{X[n, 0].max():.2f}")
        held_nodes = tabs[held]
        print(f"loading along x, sign {load_sign:+.0f}")
    print(f"grip displacement (mm) per DIC step: {np.round(grip, 2)}")

    dist_key = "distilled_mean" if args.param_mode == "mean" else "distilled_samples"
    models = ({dist_key: None} if args.distilled_dir else {}) | {args.reference: None}
    results = {}
    for key in models:
        f = os.path.join(args.out_dir, f"u_full_{key}.npz")
        if args.plot_only and os.path.exists(f):
            d = np.load(f, allow_pickle=True)
            results[key] = dict(u_dic=d["u_dic"], label=str(d["label"]), R=d["reaction_top"],
                                lo=d["u_dic_lo"] if "u_dic_lo" in d.files else None,
                                hi=d["u_dic_hi"] if "u_dic_hi" in d.files else None)
            continue
        if key == args.reference:
            mat = load_reference(key)
            label = f"{mat.label} (reference)"
            thetas, mats = [None], [mat]
        else:
            P = distilled_full_vectors(args.distilled_dir)
            if key == "distilled_mean":
                thetas = [P.mean(0)]
                label = "Distilled model (mean parameters)"
                print(f"distilled mean parameters: {describe(thetas[0])}")
            else:
                idx = np.random.default_rng(args.sample_seed).choice(len(P), min(args.n_samples, len(P)), replace=False)
                thetas = list(P[idx])
                label = f"Distilled model ({len(thetas)} parameter samples, mean field)"
            mats = [FullVectorMaterial(t) for t in thetas]
        U, R, ok = [], [], []
        for i, (mat, th) in enumerate(zip(mats, thetas)):
            tag = label if len(mats) == 1 else f"{label}: sample {i + 1}/{len(mats)}"
            try:
                u = solve(mat, X, cells, at_held, at_pulled, load_dof, load_sign, grip, tag)
                if not np.all(np.isfinite(u)):
                    raise RuntimeError("non-finite displacements")
            except Exception as exc:                                   # a failed sample is skipped
                print(f"[{tag}] FEM failed: {exc}")
                continue
            U.append(u)
            R.append(grip_reaction(mat, X, cells, u, held_nodes))
            ok.append(i)
        if not U:
            raise SystemExit(f"[{label}] no successful FEM solve")
        u_dic_s = np.stack([to_dic_nodes(u, Xd, cells, xq, Amat=Amat)[0] for u in U])     # (S, steps, m, 2)
        u_full, u_dic, R_top = np.mean(U, 0), u_dic_s.mean(0), np.mean(R, 0)
        lo = hi = None
        extra = {}
        if len(U) > 1:
            lo, hi = np.percentile(u_dic_s, 2.5, axis=0), np.percentile(u_dic_s, 97.5, axis=0)
            extra = dict(u_dic_lo=lo, u_dic_hi=hi, reaction_samples=np.array(R), u_dic_samples=u_dic_s.astype(np.float32))
            label = label.replace(f"{len(thetas)} parameter", f"{len(U)} parameter")
        params = np.array([]) if thetas[0] is None else np.array(thetas)[ok]
        np.savez_compressed(f, u_full=u_full, u_dic=u_dic, reaction_top=R_top, grip=grip, steps=steps, label=label,
                            params=params, **extra)
        print(f"[{label}] saved {f}; top-grip force at the last step {np.round(R_top[-1], 3)} (per unit thickness), "
              f"measured {np.round(R_meas[-1], 3)}")
        results[key] = dict(u_dic=u_dic, label=label, R=R_top, lo=lo, hi=hi)

    # --- plot: DIC field vs the distilled-mean model (first row) and the reference (second row)
    from plots.uq_verification_disp import plot_disp_field
    vs = np.arange(len(steps)) if args.val_steps is None else np.asarray(args.val_steps)
    k = args.plot_step
    main_key = dist_key if dist_key in results else args.reference
    main_u = results[main_key]["u_dic"]
    lo, hi = results[main_key].get("lo"), results[main_key].get("hi")
    lo_v = None if lo is None else lo[vs].reshape(-1, 2)
    hi_v = None if hi is None else hi[vs].reshape(-1, 2)
    ref = results[args.reference]
    extra = ([dict(label=ref["label"], u_step=ref["u_dic"][k], u_val_flat=ref["u_dic"][vs].reshape(-1, 2))]
             if main_key != args.reference else [])
    plot_disp_field(xq, dic_cells, u_obs[k], main_u[k], u_obs[vs].reshape(-1, 2), main_u[vs].reshape(-1, 2),
                    lo_v, hi_v, args.out_dir, filename_base="displacement_analysis_full_mesh",
                    banner_prefix="[Full specimen, displacement control]", is_clean=False, extra_rows=extra,
                    main_label=results[main_key]["label"])
    for key, r in results.items():
        print(f"{r['label']}: held-grip force per step (load direction): {np.round(np.abs(r['R'][:, load_dof]), 3)}")
    print(f"measured reactions: {np.round(R_meas[:, 0], 3)}")


if __name__ == "__main__":
    main()
