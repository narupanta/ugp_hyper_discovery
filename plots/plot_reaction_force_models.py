#!/usr/bin/env python3
"""
Reaction force at the measured displacements of each experiment: GP posterior vs distilled model vs the load cell.

No FEM solve: with the measured F of every element (DIC), the reaction of the loaded edge is the internal force summed
over its DOFs, R = sum_e B_e^T P(F_e) dA_e (the operator of the GP likelihood). Incompressible plane stress: P is the
gradient of the reduced energy Psi(F_2D, 1 / det F_2D).
  GP:        exact posterior of R (core.hsgp_inference.reaction_prediction): mean and epistemic variance.
  Distilled: R of each parameter sample (mean, 2.5 / 97.5 percentiles).
Predictive bands add the load-cell reading noise sigma_R = max(noise_rel |R_meas|, noise_floor) (the likelihood's
reaction noise: GP in quadrature, distilled as noise on each sample). Coverage = share of measured steps inside the
95% predictive band, also on the held-out steps (not in the GP training steps).

x axis: displacement between the two end cuts of the DIC window (--xaxis step: the DIC step number; the PS recording
skips frames between 5570 and 5580, so on a step axis its last point jumps).

Usage:
  python3 plots/plot_reaction_force_models.py --extract_dir results/<exp>/1/extracted --distilled_dir results/<exp>/1/distilled \
      --datasets TTf=dataset/preprocessed/experiment/ttf_20_steps_ps_incomp.npz \
                 UT=dataset/preprocessed/experiment/ut_20_steps_ps_incomp.npz \
                 PS=dataset/preprocessed/experiment/ps_20_steps_ps_incomp.npz
Writes <out_dir>/reaction_force_models.pdf/.png and reaction_force_models.json (default out_dir: the distilled dir).
"""
import argparse
import json
import os
import sys

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

jax.config.update("jax_enable_x64", True)
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from plots.theme import apply_style  # noqa: E402

EXP = "dataset/preprocessed/experiment"
DEFAULT_DATASETS = [f"TTf={EXP}/ttf_20_steps_ps_incomp.npz", f"UT={EXP}/ut_20_steps_ps_incomp.npz",
                    f"PS={EXP}/ps_20_steps_ps_incomp.npz"]
ALL_DATASETS = [f"{t}={EXP}/{t.lower()}_20_steps_ps_incomp.npz" for t in ("TTa", "TTb", "TTc", "TTd", "TTe", "TTf", "UT", "PS")]
DENSE_DATASETS = [f"{t}={EXP}/{t.lower()}_dense_ps_incomp.npz" for t in ("TTa", "TTb", "TTc", "TTd", "TTe", "TTf", "UT", "PS")]


def distilled_reactions(psi, theta, prep, batch=32):
    """(S, T) x-reaction of the loaded edge (node_type[:, 3]) for every parameter sample at the measured F."""
    F2 = jnp.asarray(prep["F"], dtype=jnp.float64)                          # (T, C, 2, 2)
    cells, dNdX, dA = jnp.asarray(prep["cells"]), jnp.asarray(prep["dNdX"]), jnp.asarray(prep["dA"])
    loaded = jnp.asarray(np.asarray(prep["node_type"])[:, 3] == 1)
    n_nodes = int(np.asarray(prep["node_type"]).shape[0])

    def P_red(th, f2):
        def psi_red(f):
            return psi(th, jnp.zeros((3, 3)).at[:2, :2].set(f).at[2, 2].set(1.0 / jnp.linalg.det(f)))
        return jax.grad(psi_red)(f2)

    @jax.jit
    def R_step(th, f_t):
        P = jax.vmap(lambda f: P_red(th, f))(f_t)                              # (C, 2, 2)
        f_el = jnp.einsum("cij,caj->cai", P, dNdX) * dA[:, None, None]         # (C, 3, 2) element nodal forces
        f_int = jnp.zeros((n_nodes, 2)).at[cells.reshape(-1)].add(f_el.reshape(-1, 2))
        return jnp.sum(jnp.where(loaded, f_int[:, 0], 0.0))

    out = []
    for b in range(0, len(theta), batch):
        tb = jnp.asarray(theta[b:b + batch])
        out.append(np.stack([np.asarray(jax.vmap(lambda th: R_step(th, F2[t]))(tb)) for t in range(F2.shape[0])], 1))
    return np.concatenate(out, 0)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--extract_dir", required=True)
    p.add_argument("--distilled_dir", default=None, help="omit: GP only")
    p.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS, help="NAME=path.npz (pipeline format)")
    p.add_argument("--all", action="store_true", help="all eight tests (TTa-TTf, UT, PS): reaction_force_models_all.pdf")
    p.add_argument("--n_samples", type=int, default=256)
    p.add_argument("--noise_rel", type=float, default=0.005, help="relative load-cell reading noise")
    p.add_argument("--noise_floor", type=float, default=0.02, help="noise floor in the data units (force per thickness)")
    p.add_argument("--xaxis", choices=["frame", "displacement", "grip", "step"], default="displacement",
                   help="x axis: displacement between the DIC end cuts (default; continuous where the recording skips "
                        "frames, e.g. PS 5570 -> 5580), the estimated grip displacement (--grip_mm_per_step x DIC step, "
                        "as the paper's force-displacement plots), the DIC step number or the frame number "
                        "(= DIC step / --step_per_frame; no assumption on the loading rate)")
    p.add_argument("--step_per_frame", type=float, default=10.0, help="--xaxis frame: DIC steps per recorded frame")
    p.add_argument("--grip_mm_per_step", type=float, default=100.0 / 6000.0,
                   help="--xaxis grip: crosshead displacement per DIC step (default: 100 mm at step 6000, constant rate)")
    p.add_argument("--dense", action="store_true",
                   help="with --all: every 100th DIC frame (dataset/preprocessed/experiment/<test>_dense_ps_incomp.npz)")
    p.add_argument("--thickness", type=float, default=None,
                   help="specimen thickness (mm): the data reactions are force per unit thickness (N/mm); with it, all "
                        "forces are shown as the total force in N (x thickness); default: N/mm")
    p.add_argument("--out_dir", default=None)
    p.add_argument("--figsize", type=float, nargs=2, default=None, help="width height in inches (default: A4 width x 30%% of A4 height)")
    p.add_argument("--show_reference", action="store_true", help="also draw the reference model curves")
    p.add_argument("--reference", default="euclid_ttf_ut",
                   help="reference model drawn on top (keys of plots/plot_reference_energy.REFERENCES; '' = none)")
    p.add_argument("--no_cache", action="store_true", help="recompute the model reactions (cache: <out_dir>/reaction_cache)")
    args = p.parse_args()
    apply_style()
    from core.hsgp_inference import reaction_prediction
    from core.teacher import load_hsgp_teacher
    from core.utils import load_f3x3_from_dataset
    gp = load_hsgp_teacher(args.extract_dir)
    with open(os.path.join(args.extract_dir, "extraction_metrics.json")) as f:
        train_steps = [int(t) for t in json.load(f).get("train_steps", [])]
    trained = set()
    cfg_p = os.path.join(args.extract_dir, "config.json")
    if os.path.exists(cfg_p):
        with open(cfg_p) as f:
            cfg = json.load(f)
        trained = {e.get("name") for e in (cfg.get("experiments") or []) if e.get("name")}
    psi = theta = None
    if args.distilled_dir:
        from plots.plot_ut_ps_global import distilled_energy_fn
        psi, theta = distilled_energy_fn(args.distilled_dir)
        theta = theta[np.random.default_rng(0).choice(len(theta), min(args.n_samples, len(theta)), replace=False)]
    out_dir = args.out_dir or args.distilled_dir or args.extract_dir
    os.makedirs(out_dir, exist_ok=True)

    sets = [s.split("=", 1) for s in ((DENSE_DATASETS if args.dense else ALL_DATASETS) if args.all else args.datasets)]
    # training frames by DIC step number (the GP training indices refer to each experiment's own training dataset)
    train_frames = {}
    for e in (cfg.get("experiments") or []) if os.path.exists(cfg_p) else []:
        if e.get("name") and e.get("dataset_path"):
            dp = e["dataset_path"] if os.path.isabs(e["dataset_path"]) else os.path.join(ROOT, e["dataset_path"])
            with np.load(dp) as z:
                train_frames[e["name"]] = np.asarray(z["steps"])[train_steps] if "steps" in z.files else None
    ncol = min(len(sets), 4)
    nrow = -(-len(sets) // ncol)
    # A4 page width (210 mm) x 30 % of its height (297 mm) by default; square panels
    fig_w, fig_h = (args.figsize or [210 / 25.4, 0.3 * 297 / 25.4])
    fig, axes = plt.subplots(nrow, ncol, figsize=(fig_w, fig_h), squeeze=False)
    ref = None
    if args.reference:
        from plots.plot_reference_energy import ReferenceMaterial
        ref = ReferenceMaterial(args.reference)
    for ax in axes.ravel()[len(sets):]:
        ax.axis("off")
    gp_col, dist_col = "#555555", "#009E73"
    summary = {}
    for ax, (name, path) in zip(axes.ravel(), sets):
        prep = dict(np.load(os.path.join(ROOT, path) if not os.path.isabs(path) else path, allow_pickle=True))
        t_sc = args.thickness or 1.0
        R = t_sc * np.asarray(prep["reaction_forces"], float)[:, 0]
        T = len(R)
        dic = np.asarray(prep["steps"]) if "steps" in prep else np.arange(T)
        if args.xaxis == "step":
            steps = dic
        elif args.xaxis == "frame":
            steps = dic / args.step_per_frame
        elif args.xaxis == "grip":
            steps = args.grip_mm_per_step * dic
        else:     # displacement between the two prescribed end cuts (bcx 1 / 2) of the DIC window
            u, bcx = np.asarray(prep["u_exp"] if "u_exp" in prep else prep["u"]), np.asarray(prep["bcx"]).ravel()
            steps = u[:, bcx == 2, 0].mean(1) - u[:, bcx == 1, 0].mean(1)
        if name in trained and train_frames.get(name) is not None:
            is_tr = np.isin(dic, train_frames[name])
        else:
            is_tr = np.isin(np.arange(T), train_steps) if name in trained else np.zeros(T, bool)
        sig_n = np.maximum(args.noise_rel * np.abs(R), t_sc * args.noise_floor)
        cache = os.path.join(out_dir, "reaction_cache", f"{os.path.basename(path)[:-4]}.npz")
        c = dict(np.load(cache)) if os.path.exists(cache) and not args.no_cache else {}
        if "gp_m" not in c or os.path.getmtime(cache) < os.path.getmtime(os.path.join(args.extract_dir, "hsgp_posterior.npz")):
            c = {}
            F3 = load_f3x3_from_dataset(prep)
            c["gp_m"], c["gp_cov"] = reaction_prediction(gp, F3, prep["cells"], prep["node_type"], prep["dNdX"], prep["dA"])
        m, cov = c["gp_m"], c["gp_cov"]
        g_m, g_sd = t_sc * m[:, 0], t_sc * np.sqrt(cov[:, 0, 0])
        g_sdp = np.sqrt(g_sd ** 2 + sig_n ** 2)
        res = {}

        def cover(lo, hi):
            hit = (R >= lo) & (R <= hi)
            return 100 * hit.mean(), (100 * hit[~is_tr].mean() if (~is_tr).any() else float("nan"))
        ec = cover(g_m - 1.96 * g_sdp, g_m + 1.96 * g_sdp)
        res["GP"] = dict(coverage=ec[0], coverage_heldout=ec[1], mape=100 * np.mean(np.abs(g_m - R) / np.abs(R)),
                         l2=float(100 * np.sum((g_m - R) ** 2) / np.sum(R ** 2)),
                         sd_epistemic_rel=float(np.median(g_sd / np.abs(R))))
        ax.fill_between(steps, g_m - 1.96 * g_sdp, g_m + 1.96 * g_sdp, color=gp_col, alpha=0.18, lw=0, zorder=2,
                        label="GP 95% (+ noise)")
        ax.fill_between(steps, g_m - 1.96 * g_sd, g_m + 1.96 * g_sd, color=gp_col, alpha=0.45, lw=0, zorder=2,
                        label="GP 95% (epistemic)")
        ax.plot(steps, g_m, color=gp_col, lw=1.0, zorder=4, label="GP mean")
        if psi is not None:
            if c.get("dist_R") is None or c["dist_R"].shape[0] != len(theta):
                print(f"[{name}] distilled reactions for {len(theta)} samples ({prep['cells'].shape[0]} elements, {T} steps) ...")
                c["dist_R"] = distilled_reactions(psi, theta, prep)
            Rd = t_sc * c["dist_R"]
            Rn = Rd + sig_n[None, :] * np.random.default_rng(1).standard_normal(Rd.shape)
            lo, hi = np.percentile(Rn, 2.5, 0), np.percentile(Rn, 97.5, 0)
            ec_d = cover(lo, hi)
            res["distilled"] = dict(coverage=ec_d[0], coverage_heldout=ec_d[1],
                                    mape=100 * np.mean(np.abs(Rd.mean(0) - R) / np.abs(R)),
                                    l2=float(100 * np.sum((Rd.mean(0) - R) ** 2) / np.sum(R ** 2)),
                                    sd_epistemic_rel=float(np.median(Rd.std(0) / np.abs(R))))
            ax.fill_between(steps, lo, hi, color=dist_col, alpha=0.15, lw=0, zorder=3, label="Distilled 95% (+ noise)")
            ax.fill_between(steps, np.percentile(Rd, 2.5, 0), np.percentile(Rd, 97.5, 0), color=dist_col, alpha=0.35,
                            lw=0, zorder=3, label="Distilled 95% (epistemic)")
            ax.plot(steps, Rd.mean(0), color=dist_col, lw=1.1, zorder=5, label="Distilled mean")
        if ref is not None:
            if c.get("ref_R") is None or str(c.get("ref_key", "")) != args.reference:
                c["ref_R"] = distilled_reactions(lambda th, F_: ref.psi(F_), np.zeros((1, 1)), prep)[0]
                c["ref_key"] = np.array(args.reference)
            Rr = t_sc * np.asarray(c["ref_R"])
            res[ref.label] = dict(coverage=float("nan"), coverage_heldout=float("nan"),
                                  mape=100 * np.mean(np.abs(Rr - R) / np.abs(R)), sd_epistemic_rel=0.0,
                                  l2=float(100 * np.sum((Rr - R) ** 2) / np.sum(R ** 2)))
            if args.show_reference:    # off by default: in the table only (evaluated like the other models)
                ax.plot(steps, Rr, "--", color="#c62828", lw=0.9, zorder=5, label=f"{ref.label} (reference)")
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        np.savez(cache, **{k: v for k, v in c.items() if v is not None})
        ms = 4 if T <= 25 else 1.3
        ax.plot(steps[~is_tr], R[~is_tr], "o", color="k", ms=ms, zorder=6, label="measured (held out)")
        if is_tr.any():
            ax.plot(steps[is_tr], R[is_tr], "s", color="tab:red", ms=ms + 1.4, zorder=7, label="measured (training step)", mew=0)
        ax.set_title(name, fontsize=7.5, pad=2)
        ax.set_box_aspect(1)
        ax.set_xlabel({"step": "DIC step", "frame": "Frame", "grip": r"Grip displacement $d$ (mm, est.)"}.get(
            args.xaxis, r"End displacement $d$ (mm)"), fontsize=7, labelpad=1)
        if args.xaxis in ("grip", "frame"):
            ax.set_xlim(0, None)
        if ax in axes[:, 0]:
            ax.set_ylabel(r"$R_x$ (N)" if args.thickness else r"$R_x$ (N/mm)", fontsize=7, labelpad=1)
        if ax not in axes[-1, :] and len(sets) > ncol:
            ax.set_xlabel("")
        ax.tick_params(labelsize=5.5, length=2, pad=1.5, width=0.5)
        ax.grid(False)
        for sp in ax.spines.values():
            sp.set_linewidth(0.6)
        summary[name] = res
        for k, v in res.items():
            print(f"[{name}] {k}: coverage {v['coverage']:.0f} % (held out {v['coverage_heldout']:.0f} %), "
                  f"MAPE {v['mape']:.2f} %, epistemic sd / |R| {100 * v['sd_epistemic_rel']:.2f} %")
    h, lab = [], []
    for ax in axes.ravel()[:len(sets)]:
        for x_, y_ in zip(*ax.get_legend_handles_labels()):
            if y_ not in lab:
                h.append(x_)
                lab.append(y_)
    order = ["measured (training step)", "measured (held out)", "GP mean", "GP 95% (epistemic)", "GP 95% (+ noise)",
             "Distilled mean", "Distilled 95% (epistemic)", "Distilled 95% (+ noise)"]
    hl = sorted(zip(h, lab), key=lambda z: order.index(z[1]) if z[1] in order else len(order))
    # legend in a column on the right: square panels are bounded by the height, so the spare width holds the legend
    fig.legend([x for x, _ in hl], [y.replace(" (reference)", "\n(reference)") for _, y in hl], loc="center right",
               ncol=1, fontsize=6, frameon=False, bbox_to_anchor=(1.0, 0.5), handlelength=1.6, handletextpad=0.4,
               labelspacing=0.7, borderaxespad=0.2)
    fig.tight_layout(rect=[0, 0, 0.83, 1], h_pad=0.4, w_pad=0.3, pad=0.2)
    out = os.path.join(out_dir, "reaction_force_models" + ("_all" if args.all else "") + ("_dense" if args.dense else "")
                       + ({"grip": "_grip", "frame": "_frame"}.get(args.xaxis, ""))
                       + (f"_t{args.thickness:g}mm" if args.thickness else "") + ".pdf")
    fig.savefig(out, bbox_inches="tight", dpi=200)
    fig.set_dpi(200)
    fig.canvas.print_png(out[:-4] + ".png")
    plt.close(fig)
    with open(out[:-4] + ".json", "w") as f:
        json.dump(dict(extract_dir=args.extract_dir, distilled_dir=args.distilled_dir, thickness=args.thickness,
                       noise_rel=args.noise_rel,
                       noise_floor=args.noise_floor, train_steps=train_steps, results=summary), f, indent=2)
    print(f"saved {out} (+ .png, .json)")
    write_force_table(summary, trained, out[:-4].replace("reaction_force_models", "reaction_force_metrics") + ".tex",
                      "every 10th DIC frame" if args.dense else "20 DIC steps", ref.label if ref is not None else None)


# L2 of the force-displacement curves reported by Flaschel et al. (FEM of the full specimens, fraction -> %); UT and PS:
# L2(P11) of their Table 4. Quoted for context only: their evaluation (full-specimen FEM, grip displacement) differs
# from ours (reactions from the measured DIC field), so the like-for-like comparison is EUCLID evaluated our way.
PAPER_L2_FORCE = {"TTa": 0.03, "TTb": 0.19, "TTc": 0.26, "TTd": 0.34, "TTe": 0.33, "TTf": 0.11, "UT": 0.16, "PS": 0.07}


def write_force_table(summary, trained, path, step_label="frames", ref_label=None):
    """LaTeX table (booktabs): per test the GP and the distilled model (L2, mean relative error, EC95 of the predictive
    band: held-out steps for the training tests), the reference model evaluated the same way (L2, mean relative error)
    and the L2 reported by Flaschel et al. (dagger, context); lowest L2 among the same-evaluation models in bold."""
    def f(v, fmt):
        return "--" if v is None or (isinstance(v, float) and np.isnan(v)) else fmt.format(v)
    rows = []
    for t, res in summary.items():
        keys = ["GP", "distilled"] + ([ref_label] if ref_label in res else [])
        l2 = {k: res[k].get("l2") for k in keys if k in res}
        best = min(l2, key=lambda k: l2[k])
        cells = [t + (r"$^\ast$" if t in trained else "")]
        for k in keys:
            r = res.get(k, {})
            v = f(r.get("l2"), "{:.3f}")
            cells += [r"\textbf{" + v + "}" if best == k else v, f(r.get("mape"), "{:.1f}")]
            if k != ref_label:
                cells.append(f(r.get("coverage_heldout") if t in trained else r.get("coverage"), "{:.0f}"))
        cells.append(f(PAPER_L2_FORCE.get(t), "{:.2f}") + r"$^\dagger$")
        rows.append(" & ".join(cells) + r" \\")
    has_ref = any(ref_label in r for r in summary.values())
    head_ref = r" & \multicolumn{2}{c}{EUCLID}" if has_ref else ""
    cols = "l ccc ccc" + (" cc" if has_ref else "") + " c"
    sub = r"$\mathcal{L}^2$ & $\bar{\epsilon}_{\mathrm{rel}}$ & EC$_{95}$"
    tex = "\n".join([
        r"\begin{table}[t]", r"\centering", r"\small", r"\begin{tabular}{" + cols + "}", r"\toprule",
        r" & \multicolumn{3}{c}{GP} & \multicolumn{3}{c}{UGP + Distillation}" + head_ref + r" & EUCLID \\",
        r"\cmidrule(lr){2-4} \cmidrule(lr){5-7}" + (r" \cmidrule(lr){8-9} \cmidrule(lr){10-10}" if has_ref else r" \cmidrule(lr){8-8}"),
        r"Test & " + sub + " & " + sub + (r" & $\mathcal{L}^2$ & $\bar{\epsilon}_{\mathrm{rel}}$" if has_ref else "")
        + r" & $\mathcal{L}^2$ (ref.) \\",
        r"\midrule", *rows, r"\bottomrule", r"\end{tabular}",
        r"\caption{Reaction force of all specimens computed from the measured DIC displacement fields (" + step_label
        + r"). All values in \%. $\mathcal{L}^2 = 100 \sum (F^{\mathrm{model}} - F^{\mathrm{exp}})^2 / \sum "
        r"(F^{\mathrm{exp}})^2$ as in Flaschel et al.; $\bar{\epsilon}_{\mathrm{rel}}$: mean relative error "
        r"$100\,|F^{\mathrm{model}} - F^{\mathrm{exp}}|/F^{\mathrm{exp}}$; EC$_{95}$: share of the measured steps inside "
        r"the 95\,\% predictive band (epistemic + 0.5\,\% load-cell noise). $^\ast$Training test (EC$_{95}$ on the "
        r"held-out steps). EUCLID is evaluated in the same way with its published coefficients; lowest $\mathcal{L}^2$ "
        r"among these models in bold. $^\dagger$Reported by Flaschel et al.\ for their full-specimen FEM (UT, PS: "
        r"$\mathcal{L}^2(P_{11})$), for reference only (different evaluation).}",
        r"\label{tab:reaction_force_metrics}", r"\end{table}"])
    with open(path, "w") as fh:
        fh.write(tex + "\n")
    print(f"saved {path}")
    print(tex)


if __name__ == "__main__":
    main()
