#!/usr/bin/env python3
"""
One figure for the discovered model of a distillation run (deviatoric and, if present, fibre terms):
  left:   posterior histograms of the discovered (non-pruned) parameters with mean and 95 % interval (2.5 / 97.5
          percentiles of the flow samples, values in the titles); true values in red when the data have a ground
          truth (recipe material_params);
          the mean total-order Sobol index of each term in its title;
  right:  parity of the total energy, GP and distilled mean (95 % interval as error bars) vs the ground truth at the
          validation states, with R^2 / RMSE / EC95 of both in a table below (energy_parity_total.npz, written by plots/plot_distilled_r2_energy.py), and the legends.

Usage:
  python3 plots/plot_discovery_summary.py --distilled_dir <run dir> [--unit MPa]
  bottom: energy along the six standard modes vs gamma (GP, distilled, truth), the GP extrapolation shaded from
          the first state that leaves the data region (mode_energy_g<max_gamma>.npz from plots/plot_split_summary.py)
Writes <distilled_dir>/discovery_summary.pdf/.png/.json and prints the parameter table.
"""
import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from plots.theme import apply_style  # noqa: E402

ANISO_NAMES = ["C42", "C43", "C44", "C62", "C63", "C64"]
TRUE_COL, DIST_COL, GP_COL, EXTRAP_COL = "#d62728", "#009E73", "#757575", "#E69F00"


def tex_name(p):
    return p if p.startswith("$") else f"${p}$"


def plain(p):
    return p.replace("$", "").replace("_", "").replace("{", "").replace("}", "").replace("\\", "")


def find_config(d):
    """The experiment recipe (config.yaml with material_params) in d or one of its parents."""
    p = os.path.abspath(d)
    for _ in range(6):
        f = os.path.join(p, "config.yaml")
        if os.path.exists(f):
            c = yaml.safe_load(open(f)) or {}
            if "material_params" in c:
                return c
        p = os.path.dirname(p)
    return {}


def fibre_invariants(F, angles_deg):
    """Isochoric I4, I6 of the two in-plane fibre directions at the given angles (degrees)."""
    C = np.einsum("nki,nkj->nij", F, F)
    Cb = C / np.linalg.det(F)[:, None, None] ** (2.0 / 3.0)
    out = []
    for a in list(angles_deg)[:2]:
        v = np.array([np.cos(np.radians(a)), np.sin(np.radians(a)), 0.0])
        out.append(np.einsum("i,nij,j->n", v, Cb, v))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--distilled_dir", required=True)
    ap.add_argument("--unit", default="MPa", help="unit of the parameters")
    ap.add_argument("--out_name", default="discovery_summary")
    ap.add_argument("--max_gamma", type=float, default=2.0,
                    help="standard-mode curves: mode_energy_g<max_gamma>.npz (plots/plot_split_summary.py --max_gamma)")
    ap.add_argument("--width", type=float, default=7.0, help="figure width (inches)")
    args = ap.parse_args()
    apply_style()
    from core.material_library import dev16, dev_names, distilled_library
    from plots.sensitivity import load_sensitivity_data
    d = args.distilled_dir
    cfg = find_config(d)
    mp = cfg.get("material_params") or {}
    has_truth = bool(mp) and str(cfg.get("has_ground_truth", True)).lower() != "false"
    true_dev = list(mp.get("dev_params", [])) + [0.0] * 10
    true_an = list(mp.get("aniso_params", [])) + [0.0] * 6

    # discovered terms per component, their samples and true values
    comps = {}
    for comp in ("dev", "aniso"):
        sens = os.path.join(d, "output", f"{comp}_sensitivities")
        rp = os.path.join(sens, "relevant_parameters.txt")
        if not os.path.exists(rp):
            continue
        with open(rp) as f:
            disc = [ln.strip() for ln in f if ln.strip()]
        stats = pd.read_csv(os.path.join(sens, "total_sobol_indices_statistics_output_0.csv"), index_col=0)
        if comp == "dev":
            lib = distilled_library(d)
            smp = dev16(np.load(os.path.join(d, "dev_flow_samples.npy")), lib)
            from core.material_library import dev16_names, distilled_gmr_order
            names = dev16_names(distilled_gmr_order(d))     # dev16 layout: [C10..E | mu1, alpha1, ... | C40 ...]
            idx = [names.index(plain(p)) for p in disc]
            # an Ogden term mu_k is shown with its exponent alpha_k when alpha_k was fitted (it varies over the samples)
            st_src, d2, i2 = {}, [], []
            for p, i in zip(disc, idx):
                d2.append(p); i2.append(i); st_src[p] = p
                if names[i].startswith("mu") and np.ptp(smp[:, i + 1]) > 1e-12:
                    k = names[i][2:]
                    d2.append(rf"$\alpha_{{{k}}}$"); i2.append(i + 1); st_src[d2[-1]] = p
            disc, idx = d2, i2
            # dev16 index -> true dev_params index: 0..9 as is, 26 + k (GMR order > 3) -> 10 + k, Ogden: no GMR truth
            tv = [true_dev[i] if i < 10 else (true_dev[i - 16] if i >= 26 and i - 16 < len(true_dev) else 0.0)
                  for i in idx]
        else:
            smp = np.load(os.path.join(d, "aniso_flow_samples.npy"))
            idx = [ANISO_NAMES.index(plain(p)) for p in disc]
            tv = [true_an[i] for i in idx]
        comps[comp] = dict(disc=disc, X=smp[:, idx], true=tv, stats=stats,
                           st_src=st_src if comp == "dev" else {p: p for p in disc})
    discovered = [p for c in comps.values() for p in c["disc"]]
    X = np.concatenate([c["X"] for c in comps.values()], axis=1)
    truth = np.array([t for c in comps.values() for t in c["true"]], float)
    nd = len(discovered)
    expo = [int(np.floor(np.log10(np.abs(X[:, i]).mean()))) if 0 < np.abs(X[:, i]).mean() < 1e-2 else 0
            for i in range(nd)]
    sc = 10.0 ** np.array(expo)
    Xs, Ts = X / sc, truth / sc
    unit = "" if has_truth else args.unit                # synthetic data: dimensionless parameters, no unit
    units = ["" if plain(p).startswith("alpha") else unit for p in discovered]    # Ogden exponents: dimensionless
    lab = [tex_name(p) + (rf" ($\times 10^{{{e}}}$" + (f" {u})" if u else ")") if e else (f" ({u})" if u else ""))
           for p, e, u in zip(discovered, expo, units)]
    rho = np.corrcoef(X.T) if nd > 1 else np.ones((1, 1))
    print(f"{'parameter':10s} {'mean':>12s} {'2.5%':>12s} {'97.5%':>12s} {'true':>8s}  ({args.unit})")
    for i, p in enumerate(discovered):
        m, lo, hi = X[:, i].mean(), *np.percentile(X[:, i], [2.5, 97.5])
        print(f"{plain(p):10s} {m:12.4e} {lo:12.4e} {hi:12.4e} {truth[i]:8.3g}")
    sc_comp = next((c for c in ("dev", "aniso") if c in comps and len(comps[c]["disc"]) >= 2), None)

    # ---- layout (inches): one grid of square panels, 6 per row
    #      top:    posterior histograms (slots 0..nd-1) | energy parity (last slot)
    #      bottom: energy of the six standard modes vs gamma (GP, distilled, truth; extrapolation shaded)
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    pal = plt.cm.tab10(np.arange(10))
    col_of = {p: pal[i % 10] for i, p in enumerate(discovered)}
    st_of = {p: float(c["stats"].loc["mean", c["st_src"][p]]) for c in comps.values() for p in c["disc"]}
    fs, fs_tick, fs_small = 7.0, 6.0, 5.6
    W = args.width
    nslot = 6
    # histogram grid: one row, or two rows with more than five terms (Ogden pairs: mu_k above alpha_k, one column
    # per term; otherwise filled row by row)
    if nd > 5 and all(plain(p).startswith(("mu", "alpha")) for p in discovered) and \
            [plain(p)[:2] for p in discovered[::2]] == ["mu"] * ((nd + 1) // 2) and nd % 2 == 0:
        pos = [(i % 2, i // 2) for i in range(nd)]
    elif nd > 5:
        pos = [(i // int(np.ceil(nd / 2)), i % int(np.ceil(nd / 2))) for i in range(nd)]
    else:
        pos = [(0, i) for i in range(nd)]
    nrow_h, ncol_h = max(r for r, _ in pos) + 1, max(c for _, c in pos) + 1
    m_l, m_r, gm = 0.40, 0.06, 0.30
    pm = (W - m_l - m_r - (nslot - 1) * gm) / nslot              # panel size
    t_top, xl_top, g_rows, t_bot, xl_bot, b_leg = 0.46, 0.52, 0.04, 0.20, 0.32, 0.30
    # top row: histograms | S_T colour bar | [parity panel + metrics box], the block centred in the space left;
    # with many terms the histograms shrink so that everything fits in one row
    cb_room, box_w = 0.48, 0.95
    block_w = 0.42 + pm + 0.10 + box_w                     # parity y label + ticks, panel, gap, metrics boxes
    hc, g_top = pm, gm
    if m_l + ncol_h * pm + (ncol_h - 1) * gm + cb_room + block_w + m_r > W:
        g_top = 0.14 if nrow_h == 1 else 0.22
        hc = (W - m_l - m_r - cb_room - block_w - (ncol_h - 1) * g_top) / ncol_h
    if nrow_h > 1:                                         # two histogram rows: smaller panels, figure stays compact
        hc, g_top = min(hc, 0.68 * pm), min(g_top, 0.22)
    x_cb = m_l + ncol_h * hc + (ncol_h - 1) * g_top + 0.18
    x_free = x_cb + cb_room - 0.18                         # left edge of the free space
    x_par = x_free + max(0.0, (W - m_r - x_free - block_w) / 2) + 0.42

    t_top = max(0.46 - (pm - hc), 0.12)                   # smaller histograms: less room above
    # histogram colour: one colormap, the shade set by the mean Sobol index (log scale, pruning threshold to 1)
    from matplotlib.colors import LinearSegmentedColormap, LogNorm
    st_cmap = LinearSegmentedColormap.from_list("st", plt.cm.Blues(np.linspace(0.06, 1.0, 256)))   # full contrast
    st_norm = LogNorm(vmin=1e-4, vmax=1.0)
    me = os.path.join(d, f"mode_energy_g{args.max_gamma:g}.npz")
    has_modes = os.path.exists(me)
    xl_in = 0.30                                           # x label of an upper histogram row
    top_h = nrow_h * (t_top + hc) + (nrow_h - 1) * xl_in + xl_top
    H = top_h + (g_rows + t_bot + pm + xl_bot if has_modes else 0.0) + b_leg
    fig = plt.figure(figsize=(W, H))
    rng = np.random.default_rng(0)
    y_hist = [H - t_top - hc - r * (hc + xl_in + t_top) for r in range(nrow_h)]
    y_row1 = (y_hist[0] + hc + y_hist[-1]) / 2 - pm / 2     # parity panel: centred on the histogram block
    y_row2 = b_leg + xl_bot

    def rect(x, y, w, h):
        return [x / W, y / H, w / W, h / H]

    def slot(j):
        return m_l + j * (pm + gm)

    def style(ax):
        ax.tick_params(labelsize=fs_tick, length=2, pad=1.2, width=0.5)
        ax.grid(False)
        for sp in ax.spines.values():
            sp.set_linewidth(0.5)

    # posterior histograms; title: True | Mean, 95% CI, mean Sobol index S_T
    for i, p in enumerate(discovered):
        r_, c_ = pos[i]
        ax = fig.add_axes(rect(m_l + c_ * (hc + g_top), y_hist[r_], hc, hc))
        style(ax)
        x = Xs[:, i]
        m, lo, hi = x.mean(), *np.percentile(x, [2.5, 97.5])
        cc = st_cmap(st_norm(np.clip(st_of[p], 1e-4, 1.0)))
        ax.hist(x, bins=30, color=cc, edgecolor="white", linewidth=0.2)
        ax.axvspan(lo, hi, color=cc, alpha=0.12, lw=0)
        ax.axvline(m, color="k", lw=0.7)
        for v in (lo, hi):
            ax.axvline(v, color="k", lw=0.5, ls="--")
        if has_truth:
            ax.axvline(Ts[i], color=TRUE_COL, lw=1.1)
            xl = ax.get_xlim()
            pad = 0.06 * (xl[1] - xl[0])
            ax.set_xlim(min(xl[0], Ts[i] - pad), max(xl[1], Ts[i] + pad))
        s_ = st_of[p]
        fst = fs_small - (0.5 if nd > 3 else 0)
        lh = 0.085 / hc                                    # line height (axes fraction)
        lines = ([(f"True: {Ts[i]:.3f}", TRUE_COL)] if has_truth else []) + \
                [(f"Mean: {m:.3f}", "k"), ((f"95% CI: " if hc >= 0.8 else "") + f"[{lo:.3f}, {hi:.3f}]", "k")]
        for j, (txt, cc_) in enumerate(lines[::-1]):      # three lines above the panel, "True" in red on top
            ax.text(0.5, 1.03 + j * lh, txt, transform=ax.transAxes, ha="center", va="bottom", fontsize=fst,
                    color=cc_)
        ax.set_yticks([])
        ax.xaxis.set_major_locator(plt.MaxNLocator(3 if hc >= 0.8 else 2))
        ax.set_xlabel(lab[i], fontsize=fs, labelpad=1.2)

    # energy parity in the last slot: GP (grey) and distilled (green) vs the truth, R^2 and EC95 inside
    pf = os.path.join(d, "energy_parity_total.npz")
    parity, ref_lab = None, None
    if os.path.exists(pf):
        z = np.load(pf)
        ref_lab = str(z["ref_label"]) if "ref_label" in z.files else None    # experimental: reference model
        t = z["true"]
        mods = {"GP": (z["gp_mean"], z["gp_mean"] - 1.96 * z["gp_std"], z["gp_mean"] + 1.96 * z["gp_std"], GP_COL),
                "Distilled": (z["dist_mean"], z["dist_q025"], z["dist_q975"], DIST_COL)}
        ax = fig.add_axes(rect(x_par, y_row1, pm, pm))
        k = rng.choice(len(t), min(400, len(t)), replace=False)
        for zo, (name, (mu, lo, hi, cc)) in enumerate(mods.items()):
            ax.errorbar(t[k], mu[k], yerr=[mu[k] - lo[k], hi[k] - mu[k]], fmt="none", ecolor=cc, alpha=0.3,
                        elinewidth=0.4, zorder=1 + 2 * zo)
            ax.scatter(t, mu, s=0.8, color=cc, alpha=0.5, lw=0, zorder=2 + 2 * zo, rasterized=True)
        allv = np.concatenate([t] + [m_[0] for m_ in mods.values()])
        lim = [allv.min(), allv.max()]
        ax.plot(lim, lim, "k--", lw=0.5, zorder=6)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        parity = {}
        for j, (n, (mu, lo, hi, cc)) in enumerate(mods.items()):
            parity[n] = dict(r2=float(1 - np.sum((t - mu) ** 2) / np.sum((t - t.mean()) ** 2)),
                             rmse=float(np.sqrt(np.mean((t - mu) ** 2))),
                             rel_err=float(100 * np.mean(np.abs(mu - t) / (np.abs(t) + 1e-8))),
                             ec95=float(100 * np.mean((t >= lo) & (t <= hi))))
            print(f"energy parity {n}: R2 {parity[n]['r2']:.4f}, RMSE {parity[n]['rmse']:.4f}, EC95 {parity[n]['ec95']:.1f} %")
        parity["n_states"] = int(len(t))
        # metrics right of the parity panel: one rounded box per model (bold title, edge in the model colour),
        # in the style of the distilled validation plots
        bx = fig.add_axes(rect(x_par + pm + 0.10, y_row1, box_w, pm))
        bx.axis("off")
        titles = {"GP": "GP Extraction", "Distilled": "Distilled"}
        for j, (n, (_, _, _, cc)) in enumerate(mods.items()):
            txt = (r"$\bf{" + titles[n].replace(" ", r"\ ") + "}$\n"
                   + r"$R^2$" + f": {parity[n]['r2']:.4f}\n"
                   + r"$\bar{\epsilon}_{\mathrm{rel}}$" + f": {parity[n]['rel_err']:.1f}%\n"
                   + r"EC$_{95\%}$" + f": {parity[n]['ec95']:.1f}%")
            bx.text(0.02, 0.98 - 0.52 * j, txt, transform=bx.transAxes, ha="left", va="top", fontsize=fs_small,
                    linespacing=1.15,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=cc, lw=1.0))
        ax.set_xlabel(r"$\Psi_{\mathrm{" + (ref_lab or "true") + "}}$", fontsize=fs, labelpad=1.2)
        ax.set_ylabel(r"$\Psi_{\mathrm{pred}}$", fontsize=fs, labelpad=1)
        ax.xaxis.set_major_locator(plt.MaxNLocator(3))
        ax.yaxis.set_major_locator(plt.MaxNLocator(3))
        style(ax)

    # energy along the standard modes vs gamma; extrapolation from the first state that leaves the data region
    if has_modes:
        zm = np.load(me)
        names = [str(n) for n in zm["mode_names"]]
        long = {"UT": "uniaxial tension", "ET": "equibiaxial tension", "PS": "pure shear",
                "UC": "uniaxial compression", "EC": "equibiaxial compression", "SS": "simple shear"}
        xv = np.asarray(zm["gamma"])
        for i, n in enumerate(names):
            ax = fig.add_axes(rect(slot(i), y_row2, pm, pm))
            out = np.where(~zm["inside"][i][1:])[0]
            if len(out):
                xb = float(xv[out[0] + 1])
                ax.axvspan(xb, xv.max(), color=EXTRAP_COL, alpha=0.18, lw=0, zorder=0)
                ax.axvline(xb, color=EXTRAP_COL, lw=0.8, ls=":", zorder=0)
            ax.fill_between(xv, zm["gp_mean"][i] - 1.96 * zm["gp_std"][i], zm["gp_mean"][i] + 1.96 * zm["gp_std"][i],
                            color=GP_COL, alpha=0.25, lw=0, zorder=1)
            ax.plot(xv, zm["gp_mean"][i], color=GP_COL, lw=0.9, ls=":", zorder=3)
            ax.fill_between(xv, zm["dist_q025"][i], zm["dist_q975"][i], color=DIST_COL, alpha=0.2, lw=0, zorder=2)
            ax.plot(xv, zm["dist_mean"][i], color=DIST_COL, lw=1.0, zorder=4)
            if zm["psi_true"].size:
                ax.plot(xv, zm["psi_true"][i], "k--", lw=0.8, dashes=(3, 2), zorder=5)
                y1 = 1.25 * float(np.max(zm["psi_true"][i]))
                ax.set_ylim(-0.05 * y1, y1)
            ax.set_xlim(xv.min(), xv.max())
            ax.set_title(long.get(n, n), fontsize=fs_small + 0.2, pad=2)
            ax.set_xlabel(r"$\gamma$", fontsize=fs, labelpad=1)
            if i == 0:
                ax.set_ylabel(r"$\Psi$", fontsize=fs, labelpad=1.5)
            ax.xaxis.set_major_locator(plt.MaxNLocator(3))
            ax.yaxis.set_major_locator(plt.MaxNLocator(3))
            style(ax)

    # two legends: the parameter entries under the histograms, the energy entries at the bottom
    h_par = ([Line2D([], [], color=TRUE_COL, lw=1.1, label="true parameter")] if has_truth else []) + \
            [Line2D([], [], color="k", lw=0.7, label="posterior mean"),
             Line2D([], [], color="k", lw=0.5, ls=(0, (2, 1.5)), label="95% CI")]
    handles = []
    if has_truth:
        handles.append(Line2D([], [], color="k", lw=0.8, ls="--", label=r"true $\Psi$"))
    elif parity is not None and ref_lab:
        handles.append(Line2D([], [], color="k", lw=0.8, ls="--", label=rf"{ref_lab} $\Psi$ (reference)"))
    handles += [Patch(color=GP_COL, alpha=0.5, label="GP (mean, 95%)"),
                Patch(color=DIST_COL, alpha=0.6, label="distilled (mean, 95%)")]
    if has_modes:
        handles.append(Patch(color=EXTRAP_COL, alpha=0.3, label="GP extrapolation"))
    x_hc = (m_l + (ncol_h * hc + (ncol_h - 1) * g_top) / 2) / W        # centre of the histogram block
    fig.legend(handles=h_par, loc="upper center", bbox_to_anchor=(x_hc, (y_hist[-1] - 0.34) / H), ncol=len(h_par),
               fontsize=fs_small, frameon=False, handlelength=1.4, columnspacing=1.0, handletextpad=0.3,
               borderaxespad=0)
    # colour bar of the mean Sobol index (histogram shades) at the right of the legend strip
    cax = fig.add_axes(rect(x_cb, y_hist[-1], 0.06, y_hist[0] + hc - y_hist[-1]))   # thin bar right of the histograms
    from matplotlib.cm import ScalarMappable
    cb = fig.colorbar(ScalarMappable(norm=st_norm, cmap=st_cmap), cax=cax, orientation="vertical")
    cb.set_ticks([])
    cb.ax.minorticks_off()
    cax.text(0.5, 1.02, r"$\bar{S}_T$" + "\nhigh", transform=cax.transAxes, ha="center", va="bottom",
             fontsize=fs_small - 0.8, linespacing=1.0)
    cax.text(0.5, -0.02, "low", transform=cax.transAxes, ha="center", va="top", fontsize=fs_small - 0.8)
    cb.outline.set_linewidth(0.4)
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.05 / H), ncol=len(handles),
               fontsize=fs_small, frameon=False, handlelength=1.4, columnspacing=0.8, handletextpad=0.3,
               borderaxespad=0)

    out = os.path.join(d, args.out_name)
    fig.savefig(out + ".pdf", dpi=300)
    fig.set_dpi(250)
    fig.canvas.print_png(out + ".png")
    plt.close(fig)
    print(f"figure {W * 25.4:.0f} x {H * 25.4:.0f} mm; saved {out}.pdf/.png")
    with open(out + ".json", "w") as f:
        json.dump(dict(parameters={plain(p): dict(mean=float(X[:, i].mean()),
                                                  q025=float(np.percentile(X[:, i], 2.5)),
                                                  q975=float(np.percentile(X[:, i], 97.5)),
                                                  true=float(truth[i]) if has_truth else None)
                                   for i, p in enumerate(discovered)},
                       correlation={f"{plain(discovered[i])}-{plain(discovered[j])}": float(rho[i, j])
                                    for i in range(nd) for j in range(i + 1, nd)},
                       energy_parity=parity), f, indent=1)


if __name__ == "__main__":
    main()
