#!/usr/bin/env python3
"""
Noise-threshold sweep of the HSGP extraction: for each displacement-noise level and noise seed, run the extraction
(scripts/run_hsgp_pipeline.sh, extraction + truth plots) from one base recipe and collect the energy and stress
metrics on the validation steps; report up to which noise level the extraction still works.

Free-residual noise sigma_f across the levels (--force_noise_rule):
  scaled  (default)  sigma_f = force_noise_scale x disp_noise. The free residual of the true material on noisy data
                     scales linearly with the displacement noise, so this keeps the same relative weighting at every
                     level (1000 x 1e-5 = 0.01, the setting that recovers the Isihara energy at 1e-5).
  fixed              sigma_f = --force_noise at every level.
  derived            hsgp_force_noise 0: sigma_f from disp_noise and a linear-elastic tangent.
  evidence           per-step sigma_f,t = s |R_t| with s maximising the evidence of the reaction-weighted likelihood
                     (--reaction_weight: 'euclid' = N_free / N_BC (default), or a number).
  from_reaction      sigma_f,t,d = (N_BC / N_free) sigma_R,t,d (force-residual noise tied to the reaction noise).
  joint              per-step sigma_f,t = s |R_t| with log s optimised by gradient together with the kernel
                     hyperparameters on the evidence of the reaction-weighted likelihood (one solve; --reaction_weight).
  joint_refit        as joint, but the weighted evidence only selects s; with s fixed the hyperparameters and the
                     posterior are recomputed with reaction weight 1 (reactions at their actual noise level).
  em                 sigma_f inferred per direction by EM from the expected free residual (hsgp_force_noise -1,
                     residual_exact), with the reaction term weighted by --reaction_weight (as for 'evidence').
  cv                 per-step sigma_f,t = s |R_t| with s chosen by cross-validation over the training steps
                     (held-out reactions; residual_exact). The chosen s is reported per seed.
Load noise: --load_noise_levels gives a grid displacement x load noise (data generated with each load_noise); the
likelihood's reactions use sigma_R = load_noise x |reading| (matched), or --reaction_noise_rel if given.

A level "works" when the mean over seeds of the validation energy R^2 >= --r2_min and of the validation energy
coverage (95% band) >= --ec_min; the threshold is the largest level up to which every level works.

The grid can be set in the base recipe (sweep_disp_noise, sweep_load_noise, sweep_seeds, sweep_force_noise_rule);
command-line options override it.

Usage:
  python3 validation/noise_threshold_sweep.py --base_recipe configs/recipes_v2/isihara_hsgp_residual_exact.yaml \
      --noise_levels 1e-6 3e-6 1e-5 3e-5 1e-4 3e-4 --seeds 1 2 3
  Rerunning with the same --sweep_dir reuses finished levels; --collect_only only rebuilds the summary.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time

import numpy as np
import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base_recipe", required=True)
    p.add_argument("--noise_levels", type=float, nargs="+", default=None,
                   help="displacement noise levels (default: recipe key sweep_disp_noise, else 1e-6 .. 3e-4)")
    p.add_argument("--load_noise_levels", type=float, nargs="+", default=None,
                   help="load noise levels, relative (default: recipe key sweep_load_noise, else its load_noise)")
    p.add_argument("--seeds", type=int, nargs="+", default=None,
                   help="noise seeds (default: recipe key sweep_seeds, else 1 2 3)")
    p.add_argument("--force_noise_rule", choices=["scaled", "fixed", "derived", "cv", "evidence", "from_reaction", "em", "joint", "joint_refit"],
                   default=None, help="sigma_f rule (default: recipe key sweep_force_noise_rule, else scaled)")
    p.add_argument("--force_noise_scale", type=float, default=1000.0, help="scaled: sigma_f = scale x disp_noise")
    p.add_argument("--force_noise", type=float, default=0.01, help="fixed: sigma_f")
    p.add_argument("--reaction_noise_rel", type=float, default=None,
                   help="sigma_R / |reaction reading| of the likelihood (default: the load noise level)")
    p.add_argument("--reaction_weight", default="euclid",
                   help="evidence rule: reaction log-likelihood weight, a number or 'euclid' (N_free / N_BC)")
    p.add_argument("--likelihood", default=None, help="override hsgp_likelihood (default: the base recipe's)")
    p.add_argument("--r2_min", type=float, default=0.99)
    p.add_argument("--ec_min", type=float, default=90.0)
    p.add_argument("--sweep_dir", default=None, help="default results/noise_sweep_<timestamp>")
    p.add_argument("--hsgp_args", default="", help="extra pipeline options as one string, e.g. "
                                                   "'--hsgp_num_basis_dev 8 --hsgp_num_basis_vol 8'")
    p.add_argument("--collect_only", action="store_true")
    p.add_argument("--grid_index", type=int, default=None,
                   help="run only grid point i (load-major order; e.g. $SLURM_ARRAY_TASK_ID); --list_grid shows them")
    p.add_argument("--list_grid", action="store_true", help="print the grid points with their index and exit")
    p.add_argument("--cv_tol", type=float, default=None,
                   help="cv rule: pick the smallest s within this many nats of the best CV score (default: best)")
    return p.parse_args()


def level_tag(x):
    return f"{x:.0e}".replace("+", "")


def run_level(recipe_path, seeds, extra, log_path):
    """Runs the extraction pipeline for one recipe; returns the created experiment directory."""
    cmd = ["bash", "scripts/run_hsgp_pipeline.sh", recipe_path, "--seeds", " ".join(map(str, seeds))] + extra
    with open(log_path, "w") as log:
        proc = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    text = open(log_path).read()
    m = re.search(r"Created experiment directory: (\S+)", text) or re.search(r"HSGP experiment: (\S+)", text)
    if proc.returncode != 0 or not m:
        print(f"    FAILED (see {log_path})", flush=True)
        return m.group(1) if m else None
    return m.group(1)


def collect(exp_dir, seed):
    d = os.path.join(exp_dir, str(seed), "extracted")
    em_p, pm_p = os.path.join(d, "extraction_metrics.json"), os.path.join(d, "piola_stress_metrics.json")
    if not os.path.exists(em_p):
        return None
    em = json.load(open(em_p))
    out = dict(r2_val=em.get("r2_val"), ec_val=em.get("ec_val"), rmse_val=em.get("rmse_val"),
               r2_train=em.get("r2_train"), ec_train=em.get("ec_train"),
               converged=em.get("converged"), fraction_unstable=em.get("fraction_unstable"),
               sigma_f_rel=(em.get("force_noise") or {}).get("sigma_f_rel"))
    if os.path.exists(pm_p):
        pm = json.load(open(pm_p))
        for comp in ("P11", "P22"):
            if comp in pm:
                out[f"{comp}_r2_val"] = pm[comp]["val"]["r2"]
                out[f"{comp}_ec_val"] = pm[comp]["val"]["ec"]
    return out


def main():
    args = parse_args()
    sweep = os.path.abspath(args.sweep_dir or os.path.join(ROOT, "results", time.strftime("noise_sweep_%Y%m%dT%H%M%S")))
    os.makedirs(os.path.join(sweep, "recipes"), exist_ok=True)
    os.makedirs(os.path.join(sweep, "logs"), exist_ok=True)
    os.makedirs(os.path.join(sweep, "state"), exist_ok=True)
    base = yaml.safe_load(open(args.base_recipe))
    # the grid: command line, else the recipe's sweep_* keys, else defaults
    args.noise_levels = [float(x) for x in (args.noise_levels or base.get("sweep_disp_noise")
                                            or [1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4])]
    args.load_noise_levels = [float(x) for x in (args.load_noise_levels or base.get("sweep_load_noise")
                                                 or [base.get("load_noise", 0.01)])]
    args.seeds = [int(x) for x in (args.seeds or base.get("sweep_seeds") or [1, 2, 3])]
    args.force_noise_rule = args.force_noise_rule or str(base.get("sweep_force_noise_rule", "scaled"))
    loads = args.load_noise_levels
    grid = [(load, lvl) for load in loads for lvl in args.noise_levels]
    tag_of = lambda load, lvl: f"d{level_tag(lvl)}_l{level_tag(load)}"
    if args.list_grid:
        for i, (load, lvl) in enumerate(grid):
            print(f"{i}: disp noise {lvl:.0e}, load noise {load:g}  ({tag_of(load, lvl)})")
        return
    # one state file per grid point, so that parallel tasks (Slurm array) never write the same file
    state_dir = os.path.join(sweep, "state")

    def read_state():
        st = {"levels": {}}
        legacy = os.path.join(sweep, "sweep_state.json")
        if os.path.exists(legacy):
            st["levels"].update(json.load(open(legacy)).get("levels", {}))
        for f in sorted(os.listdir(state_dir)):
            if f.endswith(".json"):
                st["levels"][f[:-5]] = json.load(open(os.path.join(state_dir, f)))
        return st
    state = read_state()
    state.update(base_recipe=os.path.abspath(args.base_recipe))
    print(f"Noise sweep -> {sweep}\n  displacement noise {args.noise_levels} x load noise {loads}, seeds {args.seeds}, "
          f"sigma_f rule {args.force_noise_rule}", flush=True)

    todo = grid if args.grid_index is None else [grid[args.grid_index]]
    for load, lvl in todo:
        tag = tag_of(load, lvl)
        entry = state["levels"].get(tag, {})
        exp = entry.get("experiment_dir")
        done = exp and all(collect(exp, s) is not None for s in args.seeds)
        if args.collect_only or done:
            continue
        rec = {k: v for k, v in base.items() if not k.startswith("sweep_")}
        rec["disp_noise"], rec["load_noise"] = float(lvl), float(load)
        rec["seeds"] = list(args.seeds)
        rec["reaction_noise_rel"] = float(args.reaction_noise_rel if args.reaction_noise_rel is not None else load)
        if args.likelihood:
            rec["hsgp_likelihood"] = args.likelihood
        rule = args.force_noise_rule
        rec["hsgp_force_noise"] = {"scaled": args.force_noise_scale * lvl, "fixed": args.force_noise,
                                   "em": -1.0}.get(rule, 0.0)
        rec["hsgp_force_noise_cv"] = int(rule == "cv")
        rec["hsgp_force_noise_evidence"] = int(rule == "evidence")
        rec["hsgp_force_noise_from_reaction"] = int(rule == "from_reaction")
        rec["hsgp_force_noise_joint"] = int(rule in ("joint", "joint_refit"))
        rec["hsgp_force_noise_joint_refit"] = int(rule == "joint_refit")
        if rule == "cv" and args.cv_tol is not None:
            rec["hsgp_cv_tol"] = float(args.cv_tol)
        if rule in ("evidence", "em", "joint", "joint_refit"):
            if str(args.reaction_weight) == "euclid":
                rec["hsgp_reaction_weight_rule"] = "euclid"
            else:
                rec["hsgp_reaction_weight"] = float(args.reaction_weight)
        rp = os.path.join(sweep, "recipes", f"noise_{tag}.yaml")
        yaml.safe_dump(rec, open(rp, "w"), sort_keys=False)
        how = {"cv": "by cross-validation", "evidence": "by evidence", "from_reaction": "(N_BC/N_free) sigma_R",
               "derived": "from disp_noise", "em": "inferred (EM)", "joint": "optimised with the hyperparameters",
               "joint_refit": "selected with the hyperparameters, refit at weight 1"}.get(rule, rec["hsgp_force_noise"])
        print(f"  disp {lvl:.0e}, load {load:g}: sigma_f {how} ...", flush=True)
        target = exp if exp and os.path.isdir(exp) else rp   # an existing directory: only the missing seeds run
        t0 = time.time()
        exp = run_level(target, args.seeds, args.hsgp_args.split(), os.path.join(sweep, "logs", f"noise_{tag}.log")) or exp
        state["levels"][tag] = dict(noise=float(lvl), load=float(load), experiment_dir=exp, recipe=rp,
                                    sigma_f=rec["hsgp_force_noise"] if rule in ("scaled", "fixed") else rule,
                                    time_s=time.time() - t0)
        json.dump(state["levels"][tag], open(os.path.join(state_dir, f"{tag}.json"), "w"), indent=2)
        print(f"    done in {time.time() - t0:.0f}s -> {exp}", flush=True)
    if args.grid_index is not None and not args.collect_only:
        print("Grid point done; build the summary with --collect_only once all points have finished.")
        return
    state = read_state()

    # --- summary
    rows, table = [], []
    entries = [e for e in state["levels"].values() if e.get("load", loads[0]) in loads and e["noise"] in args.noise_levels]
    for e in sorted(entries, key=lambda e: (e.get("load", 0.0), e["noise"])):
        per_seed = {s: collect(e["experiment_dir"], s) for s in args.seeds if e.get("experiment_dir")}
        per_seed = {s: v for s, v in per_seed.items() if v is not None}
        for s, v in per_seed.items():
            rows.append(dict(noise=e["noise"], load=e.get("load"), seed=s, sigma_f=e.get("sigma_f"), **v))
        if not per_seed:
            continue
        agg = lambda k: [float(v[k]) for v in per_seed.values() if v.get(k) is not None]
        stat = {k: (np.mean(agg(k)), np.std(agg(k))) if agg(k) else (np.nan, np.nan)
                for k in ("r2_val", "ec_val", "P11_r2_val", "P11_ec_val", "fraction_unstable")}
        works = stat["r2_val"][0] >= args.r2_min and stat["ec_val"][0] >= args.ec_min
        table.append(dict(noise=e["noise"], load=e.get("load"), n=len(per_seed), works=bool(works),
                          sigma_f=e.get("sigma_f"), **{k: v[0] for k, v in stat.items()},
                          **{k + "_std": v[1] for k, v in stat.items()}))
    thresholds = {}
    for load in loads:
        thr = None
        for t in [t for t in table if t["load"] == load]:
            if not t["works"]:
                break
            thr = t["noise"]
        thresholds[load] = thr
    summary = dict(sweep_dir=sweep, criteria=dict(r2_min=args.r2_min, ec_min=args.ec_min),
                   thresholds={f"{k:g}": v for k, v in thresholds.items()}, levels=table,
                   base_recipe=os.path.abspath(args.base_recipe), force_noise_rule=args.force_noise_rule)
    json.dump(summary, open(os.path.join(sweep, "sweep_summary.json"), "w"), indent=2, default=float)
    if rows:
        import csv
        with open(os.path.join(sweep, "sweep_per_seed.csv"), "w", newline="") as f:
            keys = sorted({k for r in rows for k in r},
                          key=lambda k: (k not in ("load", "noise", "seed", "sigma_f"), k))
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
    lines = [f"# Noise-threshold sweep ({os.path.basename(os.path.abspath(args.base_recipe))})", "",
             f"sigma_f rule: {args.force_noise_rule}" + (f" (sigma_f = {args.force_noise_scale:g} x disp_noise)"
                                                         if args.force_noise_rule == "scaled" else "")
             + "; reactions: sigma_R = " + (f"{args.reaction_noise_rel:g}" if args.reaction_noise_rel is not None
                                             else "load noise") + f" x |reading|; seeds {args.seeds}.",
             f"Works: mean validation energy R^2 >= {args.r2_min} and energy coverage >= {args.ec_min}%.", "",
             "| load noise | disp noise | seeds | energy R^2 (val) | energy EC (val, %) | P11 R^2 (val) | P11 EC (val, %) | unstable | works |",
             "|---|---|---|---|---|---|---|---|---|"]
    for t in table:
        lines.append(f"| {t['load']:g} | {t['noise']:.0e} | {t['n']} | "
                     f"{t['r2_val']:.4f} ± {t['r2_val_std']:.4f} | {t['ec_val']:.1f} ± {t['ec_val_std']:.1f} | "
                     f"{t['P11_r2_val']:.4f} | {t['P11_ec_val']:.1f} | {t['fraction_unstable']:.3f} | "
                     f"{'yes' if t['works'] else 'no'} |")
    lines.append("")
    for load, thr in thresholds.items():
        lines.append(f"**Load noise {load:g}: threshold {thr:.0e}**" if thr is not None
                     else f"**Load noise {load:g}: no level meets the criteria.**")
    open(os.path.join(sweep, "sweep_summary.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))

    if table:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        for i, load in enumerate(loads):
            sub = [t for t in table if t["load"] == load]
            if not sub:
                continue
            x = np.array([t["noise"] for t in sub])
            for a, key in ((ax[0], "r2_val"), (ax[1], "ec_val")):
                y = np.array([t[key] for t in sub]); s_ = np.array([t[key + "_std"] for t in sub])
                a.errorbar(x, y, yerr=s_, marker="o", capsize=3, color=colors[i % len(colors)],
                           label=f"load noise {load:g}")
        for a, lab, ref in ((ax[0], r"Energy $R^2$ (validation steps)", args.r2_min),
                            (ax[1], "Energy coverage of the 95% band (%)", args.ec_min)):
            a.axhline(ref, ls="--", color="gray", lw=1)
            a.set_xscale("log"); a.set_xlabel("Displacement noise std"); a.set_ylabel(lab); a.grid(alpha=0.3)
            a.legend(frameon=False)
        ax[1].axhline(95, ls=":", color="k", lw=0.8)
        fig.tight_layout()
        fig.savefig(os.path.join(sweep, "noise_threshold.pdf"))
    print(f"\nSaved {sweep}/sweep_summary.md, sweep_summary.json, sweep_per_seed.csv, noise_threshold.pdf")


if __name__ == "__main__":
    main()
