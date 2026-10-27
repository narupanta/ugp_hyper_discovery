#!/usr/bin/env python3
"""
Coverage study of the distilled material parameters over noise realisations and normalizing-flow initialisations.

For an HSGP experiment (synthetic, with a ground truth) with extractions <exp>/<s>/extracted for the noise seeds s:
  for each noise seed s and initialisation m = 0..M-1 (in this order: seed 1 init 0, seed 1 init 1, ..., last seed
  last init; dev and vol of a pair run at the same time), the distillation (dev, and vol unless deviatoric-only) is run with
  seed s + 1000 m into <exp>/coverage_study_<states>/seed<s>/init<m>/ (m = 0 uses the same seed as the standard
  pipeline run). <states> names the distillation states like the GP export folder: e.g. dataset_f_n512,
  dataset_f_random_n256, standard_interp, standard_g1.0 (--sample_mode / --num_points override the recipe);
  optional --checkpoints save the parameter samples of the final distillation stage after k Wasserstein iterations
  (the training trajectory is unchanged by the checkpoints).
Evaluation, per permutation (s, m) and checkpoint: the central `level` interval of every parameter marginal; the
permutation is covered when the true value lies in the interval for ALL true (non-zero) parameters. Estimated coverage
= #covered / (N x M) x 100. Also reported: the strict variant over all candidate parameters (true zeros must be pruned
or their interval must reach 0), per-parameter coverage, and for checkpoints the drift of the parameter means towards
the final iteration relative to the spread between initialisations.

Usage:
  python3 validation/distillation_coverage_study.py --experiment_dir results/<exp> --n_noise_seeds 1 --n_inits 5 \
      --checkpoints 2000 4000 6000 8000 10000                       # pilot
  python3 validation/distillation_coverage_study.py --experiment_dir results/<exp> --n_noise_seeds 10 --n_inits 5
  python3 validation/distillation_coverage_study.py --experiment_dir results/<exp> --sample_mode standard_interp
  --evaluate_only: recompute the summary from existing runs.   --extract: run missing HSGP extractions first.
"""
import argparse
import csv
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from core.material_library import (FULL_NAMES, full_param_vectors, is_dev_only, library_from_material_model,  # noqa: E402
                                   sort_ogden_terms, true_full_vector)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--experiment_dir", required=True)
    p.add_argument("--n_noise_seeds", type=int, default=None, help="use noise seeds 1..N")
    p.add_argument("--noise_seeds", type=int, nargs="*", default=None, help="explicit noise seeds (overrides --n_noise_seeds)")
    p.add_argument("--n_inits", type=int, default=5, help="normalizing-flow initialisations per noise seed")
    p.add_argument("--n_iterations", type=int, default=None, help="Wasserstein iterations (default: recipe value)")
    p.add_argument("--checkpoints", type=int, nargs="*", default=[], help="also evaluate after these iterations")
    p.add_argument("--level", type=float, default=0.95, help="nominal level of the central interval")
    p.add_argument("--jobs", type=int, default=2, help="(unused: dev and vol of one seed/init run together, the "
                                                         "seed/init pairs one after another)")
    p.add_argument("--extract", action="store_true", help="run missing HSGP extractions (scripts/run_hsgp_pipeline.sh)")
    p.add_argument("--evaluate_only", action="store_true", help="only evaluate existing runs")
    p.add_argument("--force", action="store_true", help="rerun distillations that already have results")
    p.add_argument("--sample_mode", default=None, help="distillation states (default: recipe sample_mode), e.g. "
                   "dataset_f, dataset_f_random, standard_interp, standard")
    p.add_argument("--num_points", type=int, default=None, help="number of distillation states (default: recipe)")
    p.add_argument("--max_gamma", type=float, default=None, help="standard modes: maximum amplitude (default: recipe)")
    p.add_argument("--study_dir", default=None,
                   help="output directory (default: <experiment_dir>/coverage_study_<states>, see above)")
    p.add_argument("--no_sensitivity", action="store_true", help="skip the Sobol pruning (quick tests only)")
    p.add_argument("--no_plots", action="store_true",
                   help="skip the post-distillation plots after each seed/init (scripts/replot_coverage_study.sh)")
    return p.parse_args()


def distill_cmd(cfg, extract_dir, out_dir, component, seed, n_iter, checkpoints):
    get = lambda k, d: cfg.get(k, d)
    cmd = [sys.executable, "distillation/distill_uqmodeldisc.py", "--saved_model_dir", extract_dir,
           "--material_model", str(get("distilled_material_model", "gmr")), "--n_iterations", str(n_iter),
           "--distill_target", str(get("distill_target", "sef_split")), "--component", component,
           "--override_out_dir", out_dir, "--sample_mode", str(get("sample_mode", "standard")),
           "--sampling_mode", str(get("distillation_sampling_mode", "cached_L")),
           "--num_points", str(get("num_points", 100)), "--num_func_samples", str(get("num_func_samples", 512)),
           "--max_gamma", str(get("max_gamma", 1.0)), "--sobol_threshold", str(get("sobol_threshold", 1e-4)),
           "--sobol_samples_factor", str(get("sobol_samples_factor", 2)), "--seed", str(seed)]
    if str(get("do_sensitivity", "False")) in ("0", "False", "false"):
        cmd.append("--no_sensitivity")
    disabled = get(f"{component}_params_disabled", [])
    if disabled:
        cmd += [f"--{component}_params_disabled"] + [str(x) for x in np.atleast_1d(disabled)]
    if checkpoints:
        cmd += ["--checkpoints"] + [str(k) for k in checkpoints]
    return cmd


def export_cmd(cfg, extract_dir):
    """The GP export that distill_uqmodeldisc.py would create on demand (same subfolder naming)."""
    mode, n_pts = str(cfg.get("sample_mode", "standard")), str(cfg.get("num_points", 100))
    gamma, target = str(cfg.get("max_gamma", 1.0)), str(cfg.get("distill_target", "sef_split"))
    if mode == "dataset_all":
        sub = "pytorch_export_dataset_all"
    elif mode.startswith("dataset_f"):
        sub = f"pytorch_export_{mode}_n{n_pts}"
    elif mode in ("standard_interp", "inducing_points"):
        sub = f"pytorch_export_{mode}"
    else:
        sub = f"pytorch_export_standard_g{gamma}"
    if target in ("sef_stress", "sef_cauchy"):
        sub += f"_{target}"
    return [sys.executable, "distillation/export_gp_to_pytorch.py", "--saved_model_dir", extract_dir,
            "--sample_mode", mode, "--num_points", n_pts, "--max_gamma", gamma, "--distill_target", target,
            "--export_subfolder", sub], os.path.join(extract_dir, sub)


def run_job(job):
    cmd, log, label = job
    os.makedirs(os.path.dirname(log), exist_ok=True)
    with open(log, "w") as f:
        ok = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT).returncode == 0
    print(f"  {label}: {'done' if ok else 'FAILED (see ' + log + ')'}", flush=True)
    return ok


def samples_path(run_dir, component, k, final_iter):
    name = f"{component}_flow_samples.npy" if (k is None or k >= final_iter) else f"{component}_flow_samples_it{k}.npy"
    return os.path.join(run_dir, name)


def evaluate(run_dir, k, final_iter, library, dev_only, truth, level):
    """Interval check of one permutation at checkpoint k: dict or None if the samples are missing."""
    p_dev = samples_path(run_dir, "dev", k, final_iter)
    if not os.path.exists(p_dev):
        return None
    dev = sort_ogden_terms(np.load(p_dev), library)
    if dev_only:
        vol = np.zeros((dev.shape[0], 3))
    else:
        p_vol = samples_path(run_dir, "vol", k, final_iter)
        if not os.path.exists(p_vol):
            return None
        vol = np.load(p_vol)
    n = min(len(dev), len(vol))
    full = full_param_vectors(dev[:n], vol[:n], library)
    t = truth[:full.shape[1]]
    a = 0.5 * (1.0 - level)
    lo, hi = np.quantile(full, a, axis=0), np.quantile(full, 1.0 - a, axis=0)
    tol = 1e-12                                                    # pruned parameters are exactly 0
    inside = (t >= lo - tol) & (t <= hi + tol)
    active = t != 0.0
    return dict(inside=inside, covered_true=bool(np.all(inside[active])), covered_all=bool(np.all(inside)),
                mean=full.mean(0), std=full.std(0), lo=lo, hi=hi)


def main():
    args = parse_args()
    exp = os.path.abspath(args.experiment_dir)
    cfg = yaml.safe_load(open(os.path.join(exp, "config.yaml")))
    seeds = args.noise_seeds if args.noise_seeds else list(range(1, (args.n_noise_seeds or len(cfg.get("seeds", [1]))) + 1))
    inits = list(range(args.n_inits))
    n_iter = args.n_iterations or int(cfg.get("dev_vol_distillation_n_iterations", cfg.get("distillation_n_iterations", 5000)))
    checkpoints = sorted({int(k) for k in args.checkpoints if 0 < int(k) <= n_iter})
    eval_points = checkpoints if checkpoints else [n_iter]
    dist_model = str(cfg.get("distilled_material_model", "gmr"))
    library, dev_only = library_from_material_model(dist_model), is_dev_only(dist_model) or \
        str(cfg.get("plane_stress_mode", "")) == "incompressible"
    truth = true_full_vector(cfg.get("material_params"))
    if truth is None:
        sys.exit("No ground truth in config.yaml (material_params): the coverage study needs a synthetic experiment.")
    names = FULL_NAMES[:13] if library == "gmr" else FULL_NAMES
    for key, val in (("sample_mode", args.sample_mode), ("num_points", args.num_points), ("max_gamma", args.max_gamma)):
        if val is not None:
            cfg[key] = val
    states = os.path.basename(export_cmd(cfg, "")[1])[len("pytorch_export_"):]   # same naming as the GP export
    study = os.path.abspath(args.study_dir) if args.study_dir else os.path.join(exp, f"coverage_study_{states}")
    if args.no_sensitivity:
        cfg["do_sensitivity"] = 0
    os.makedirs(study, exist_ok=True)
    print(f"Coverage study: {exp}\n  distillation states {states} -> {study}\n  noise seeds {seeds}, {len(inits)} initialisations, {n_iter} iterations, "
          f"checkpoints {checkpoints or 'final only'}, library {library}{' (dev only)' if dev_only else ''}", flush=True)

    # --- extractions
    missing = [s for s in seeds if not os.path.exists(os.path.join(exp, str(s), "extracted", "hsgp_posterior.npz"))]
    if missing and not args.evaluate_only:
        if not args.extract:
            sys.exit(f"Missing extractions for noise seeds {missing}; rerun with --extract.")
        print(f"Running HSGP extractions for noise seeds {missing} ...", flush=True)
        subprocess.run(["bash", "scripts/run_hsgp_pipeline.sh", exp, "--seeds", " ".join(map(str, missing))],
                       cwd=ROOT, check=True)

    # --- distillations, in order: seed s / init m with its dev and vol distillation run together, then the next init,
    # then the next seed. A missing shared GP export is created first (dev and vol would otherwise both create it).
    if not args.evaluate_only:
        pairs = []
        for s in seeds:
            for m in inits:
                run_dir = os.path.join(study, f"seed{s}", f"init{m}")
                jobs = []
                for comp in (["dev"] if dev_only else ["dev", "vol"]):
                    if not args.force and os.path.exists(samples_path(run_dir, comp, None, n_iter)) and all(
                            os.path.exists(samples_path(run_dir, comp, k, n_iter)) for k in checkpoints):
                        continue
                    jobs.append((distill_cmd(cfg, os.path.join(exp, str(s), "extracted"), run_dir, comp, s + 1000 * m,
                                             n_iter, checkpoints),
                                 os.path.join(run_dir, f"{comp}_study.log"), f"seed {s} init {m} {comp}"))
                if jobs:
                    pairs.append((s, m, jobs))
        print(f"{sum(len(j) for _, _, j in pairs)} distillation runs to do ({len(pairs)} seed/init pairs)", flush=True)
        for i, (s, m, jobs) in enumerate(pairs, 1):
            print(f"[{i}/{len(pairs)}] seed {s} init {m}: {' + '.join(j[2].split()[-1] for j in jobs)}", flush=True)
            extract_dir = os.path.join(exp, str(s), "extracted")
            cmd, export_dir = export_cmd(cfg, extract_dir)
            if len(jobs) > 1 and not any(os.path.exists(os.path.join(export_dir, f))
                                         for f in ("mean_dev.npy", "mean_psi.npy")):
                print(f"  exporting the GP of seed {s}: {export_dir}", flush=True)
                with open(os.path.join(extract_dir, "export_study.log"), "w") as f:
                    if subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT).returncode != 0:
                        print(f"  export FAILED (see {extract_dir}/export_study.log); dev runs first", flush=True)
                        run_job(jobs[0])
                        jobs = jobs[1:]
            with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
                list(pool.map(run_job, jobs))
            if not args.no_plots:   # energy / stress / parameter plots of this pair, as after a normal distillation
                with open(os.path.join(study, f"seed{s}", f"init{m}", "plots.log"), "w") as f:
                    subprocess.run(["bash", "scripts/replot_coverage_study.sh", study, "--seeds", str(s), "--inits", str(m)],
                                   cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)

    # --- evaluation
    rows, summary = [], {"experiment": exp, "noise_seeds": seeds, "n_inits": len(inits), "level": args.level,
                         "n_iterations": n_iter, "library": library, "parameters": names,
                         "truth": truth[:len(names)].tolist(), "checkpoints": {}}
    final_means = {}
    for k in sorted(eval_points, reverse=True):          # final first: drift is measured against it
        res = {}
        for s in seeds:
            for m in inits:
                r = evaluate(os.path.join(study, f"seed{s}", f"init{m}"), k, n_iter, library, dev_only, truth, args.level)
                if r is not None:
                    res[(s, m)] = r
                    rows.append(dict(iteration=k, noise_seed=s, init=m, covered_true=int(r["covered_true"]),
                                     covered_all=int(r["covered_all"]),
                                     **{f"in_{n}": int(v) for n, v in zip(names, r["inside"])},
                                     **{f"mean_{n}": float(v) for n, v in zip(names, r["mean"])}))
        if not res:
            continue
        n_perm = len(res)
        cov_true = 100.0 * sum(r["covered_true"] for r in res.values()) / n_perm
        cov_all = 100.0 * sum(r["covered_all"] for r in res.values()) / n_perm
        per_param = {n: 100.0 * np.mean([r["inside"][i] for r in res.values()]) for i, n in enumerate(names)}
        entry = dict(n_permutations=n_perm, coverage_true_params=cov_true, coverage_all_params=cov_all,
                     per_parameter=per_param)
        # spread between initialisations (per noise seed, averaged) and drift of the means towards the final iteration
        means = {key: r["mean"] for key, r in res.items()}
        between = np.mean([np.std([means[(s, m)] for m in inits if (s, m) in means], axis=0)
                           for s in seeds if sum((s, m) in means for m in inits) > 1], axis=0) \
            if len(inits) > 1 else np.full(len(names), np.nan)
        entry["between_init_std"] = dict(zip(names, np.atleast_1d(between).tolist()))
        if k == max(eval_points):
            final_means = means
        else:
            drift = np.mean([np.abs(means[key] - final_means[key]) for key in means if key in final_means], axis=0)
            entry["mean_drift_to_final"] = dict(zip(names, drift.tolist()))
            with np.errstate(divide="ignore", invalid="ignore"):
                entry["drift_over_between_init_std"] = dict(zip(names, np.where(between > 0, drift / between, np.nan).tolist()))
        summary["checkpoints"][int(k)] = entry

    # --- outputs
    json.dump(summary, open(os.path.join(study, "coverage_summary.json"), "w"), indent=2, default=float)
    if rows:
        with open(os.path.join(study, "coverage_permutations.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(sorted(rows, key=lambda r: (r["iteration"], r["noise_seed"], r["init"])))
    active = [n for n, t in zip(names, truth) if t != 0.0]
    lines = [f"# Distillation coverage study ({os.path.basename(exp)})", "",
             f"Noise seeds {seeds} x {len(inits)} initialisations; nominal level {100 * args.level:.0f}% (central interval).",
             f"True parameters: " + ", ".join(f"{n} = {t:g}" for n, t in zip(names, truth) if t != 0.0), "",
             "| Iteration | Permutations | Coverage, true params (%) | Coverage, all params (%) | "
             + " | ".join(f"{n} (%)" for n in active) + " |",
             "|" + "---|" * (4 + len(active))]
    for k in sorted(summary["checkpoints"]):
        e = summary["checkpoints"][k]
        lines.append(f"| {k} | {e['n_permutations']} | {e['coverage_true_params']:.1f} | {e['coverage_all_params']:.1f} | "
                     + " | ".join(f"{e['per_parameter'][n]:.0f}" for n in active) + " |")
    drift_ks = [k for k in sorted(summary["checkpoints"]) if "drift_over_between_init_std" in summary["checkpoints"][k]]
    if drift_ks:
        lines += ["", "Drift of the parameter means to the final iteration / spread between initialisations "
                      "(< 1: the remaining training changes less than the initialisation does):", "",
                  "| Iteration | " + " | ".join(active) + " |", "|" + "---|" * (1 + len(active))]
        for k in drift_ks:
            d = summary["checkpoints"][k]["drift_over_between_init_std"]
            lines.append(f"| {k} | " + " | ".join(f"{d[n]:.2f}" for n in active) + " |")
    open(os.path.join(study, "coverage_summary.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nSaved {study}/coverage_summary.md, coverage_summary.json, coverage_permutations.csv")


if __name__ == "__main__":
    main()
