#!/usr/bin/env python3
"""
Boundary-force check on the measured field (virtual-fields style): for each load step, the internal nodal forces of a
material model evaluated at the MEASURED deformation (DIC mesh, measured F) are summed over the nodes of each end cut
(bcx = 1 and bcx = 2) and compared with the measured machine force. No FEM solve and no extrapolation: only the data
and the material model. The distilled model gives a mean and a 95% band (parameter samples); reference models
(e.g. EUCLID) one curve. Training and validation load steps (from the experiment's config.yaml) are marked.

Usage: python3 plots/plot_boundary_force_vfm.py --distilled_dir <seed>/distilled \
           --dataset dataset/preprocessed/experiment/ttf_20_steps_ps_incomp.npz \
           [--references euclid_ttf_ut euclid_noogden_ttf_ut] [--thickness 2.5] [--out <dir>]
Writes <out>/boundary_force_vfm.pdf/.png and boundary_force_vfm.json
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "validation"))


def find_config(start):
    import yaml
    d = os.path.abspath(start)
    while d != os.path.dirname(d):
        f = os.path.join(d, "config.yaml")
        if os.path.exists(f):
            c = yaml.safe_load(open(f)) or {}
            if "train_load_steps_indices" in c:      # the experiment config (not a distillation config)
                return c
        d = os.path.dirname(d)
    return {}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--distilled_dir", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--references", nargs="*", default=["euclid_ttf_ut", "euclid_noogden_ttf_ut"])
    p.add_argument("--thickness", type=float, default=None, help="mm: forces in N (default: N/mm)")
    p.add_argument("--n_samples", type=int, default=200)
    p.add_argument("--out", default=None, help="output folder (default: <distilled_dir>)")
    args = p.parse_args()
    import jax
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from core.material_library import distilled_library, full_param_vectors
    from core.material_models import IncompressiblePlaneStressView
    from forward_fem_full_specimen import FullVectorMaterial, load_reference
    from plots.theme import apply_style, save_figure

    ds = np.load(args.dataset, allow_pickle=True)
    cells, dNdX, dA = np.asarray(ds["cells"]), jnp.asarray(ds["dNdX"]), jnp.asarray(ds["dA"])
    F2 = jnp.asarray(np.asarray(ds["F"])[..., :2, :2])               # measured F per element and step
    n_nodes = len(ds["mesh_pos"])
    bcx = np.asarray(ds["bcx"]).ravel()
    ends = {"end 1": np.where(bcx == 1)[0], "end 2": np.where(bcx == 2)[0]}
    Rm = np.asarray(ds["reaction_forces"])
    ld = int(np.argmax(np.abs(Rm[-1])))
    Rm = np.abs(Rm[:, ld])
    steps = np.asarray(ds["steps"])
    t = args.thickness or 1.0
    cfg = find_config(args.distilled_dir)
    tr = np.asarray(cfg.get("train_load_steps_indices", []), int)
    va = np.asarray(cfg.get("val_load_steps_indices", []), int)

    def end_forces(mat):
        """(steps, 2 ends): |sum of the internal nodal forces (load component)| over each end cut."""
        psi = IncompressiblePlaneStressView(mat)._reduced_psi
        Pf = jax.vmap(jax.grad(lambda f2: psi(jnp.zeros((3, 3)).at[:2, :2].set(f2).at[2, 2].set(1.0 / jnp.linalg.det(f2)))))
        out = []
        for k in range(F2.shape[0]):
            fc = jnp.einsum("cij,caj->cai", Pf(F2[k]), dNdX) * dA[:, None, None]
            f = np.zeros((n_nodes, 2))
            np.add.at(f, cells, np.asarray(fc))
            out.append([abs(f[n, ld].sum()) for n in ends.values()])
        return t * np.array(out)

    dev = np.load(os.path.join(args.distilled_dir, "dev_flow_samples.npy"))
    full = full_param_vectors(dev, np.zeros((len(dev), 3)), distilled_library(args.distilled_dir))
    idx = np.random.default_rng(0).choice(len(full), min(args.n_samples, len(full)), replace=False)
    print(f"distilled: {len(idx)} parameter samples ...", flush=True)
    Fs = np.array([end_forces(FullVectorMaterial(full[i])) for i in idx])          # (S, steps, 2)
    F_mean = end_forces(FullVectorMaterial(full.mean(0)))
    lo, hi = np.percentile(Fs, 2.5, axis=0), np.percentile(Fs, 97.5, axis=0)
    refs = {k: end_forces(load_reference(k)) for k in args.references}
    labels = {k: load_reference(k).label.replace(" (UT+TTf)", "") for k in args.references}
    Rm_t = t * Rm

    def rel_err(Fx, sel):
        return float(100 * np.mean(np.abs(Fx[sel] - Rm_t[sel, None]) / Rm_t[sel, None])) if len(sel) else float("nan")
    summary = {"unit": "N" if args.thickness else "N/mm", "thickness": args.thickness,
               "distilled_mean": {"train_rel_err_%": rel_err(F_mean, tr), "val_rel_err_%": rel_err(F_mean, va),
                                  "val_coverage_%": float(100 * np.mean((Rm_t[va, None] >= lo[va]) & (Rm_t[va, None] <= hi[va])))
                                  if len(va) else float("nan")}}
    for k, Fx in refs.items():
        summary[k] = {"train_rel_err_%": rel_err(Fx, tr), "val_rel_err_%": rel_err(Fx, va)}

    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
    cols = ["#c62828", "#6a1b9a", "#ef6c00"]
    for j, (ax, nm) in enumerate(zip(axes, ends)):
        ax.fill_between(steps, lo[:, j], hi[:, j], color="#009E73", alpha=0.25, lw=0, label="distilled 95% band")
        ax.plot(steps, F_mean[:, j], color="#009E73", lw=2, label="distilled (mean parameters)")
        for (k, Fx), c in zip(refs.items(), cols):
            ax.plot(steps, Fx[:, j], "--", color=c, lw=1.4, label=labels[k])
        if len(tr):
            ax.plot(steps[tr], Rm_t[tr], "ko", ms=4.5, mfc="k", label="measured (training step)")
        if len(va):
            ax.plot(steps[va], Rm_t[va], "ko", ms=4.5, mfc="white", label="measured (validation step)")
        ax.set_xlabel("DIC step")
        ax.set_title(f"Sum of nodal forces at {nm} ({len(ends[nm])} nodes)", fontsize=10.5)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("reaction force (N)" if args.thickness else "reaction force per thickness (N/mm)")
    axes[1].legend(fontsize=8, loc="upper left")
    s = summary["distilled_mean"]
    txt = (f"validation steps, mean |rel. error|: distilled {s['val_rel_err_%']:.1f}% "
           f"(band covers the measurement at {s['val_coverage_%']:.0f}% of step-ends)"
           + "".join(f"; {labels[k]} {summary[k]['val_rel_err_%']:.1f}%" for k in refs))
    fig.text(0.5, 0.01, txt, ha="center", va="bottom", fontsize=8.5)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    out = args.out or args.distilled_dir
    os.makedirs(out, exist_ok=True)
    save_figure(fig, os.path.join(out, "boundary_force_vfm.pdf"))
    fig.set_dpi(200)
    fig.canvas.print_png(os.path.join(out, "boundary_force_vfm.png"))
    plt.close(fig)
    json.dump(summary, open(os.path.join(out, "boundary_force_vfm.json"), "w"), indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
