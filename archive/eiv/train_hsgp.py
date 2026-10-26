"""
ARCHIVED (EIV and all likelihood / force-noise variants; superseded by extraction/train_hsgp.py, residual_exact + CV).
Run from the project root: python3 archive/eiv/train_hsgp.py --recipe ... (recipes: git f4d94ca, configs/recipes_v2).
HSGP extraction (closed-form reduced-rank GP posterior under the EIV likelihood) for one seed.

Writes the same layout as extraction/train_unsupervised.py into --batch_dir:
    hsgp_posterior.npz, hsgp_history.json, config.json, I_obs_all.npy, extraction_metrics.json and the plots.

--phase fit|plot|all: the fit and the plots run in separate processes on small machines (peak memory adds up
otherwise); scripts/run_hsgp_pipeline.sh calls the two phases one after the other.

HSGP settings are read from the recipe (keys below) and can be overridden on the command line:
    hsgp_likelihood: eiv            # eiv | residual | residual_learned | residual_exact (needs disp_noise or hsgp_force_noise)
                                    # | nigp (exact residual warm-up, then K Sigma_u K^T input noise; needs disp_noise)
    hsgp_nigp_warmup: residual      # nigp warm-up: residual (exact residual Gaussian) | linear (linear-elastic stage only)
    hsgp_eiv_warmup: none           # eiv: none | residual_exact (first GP iteration = exact residual Gaussian with
                                    # hsgp_force_noise / disp_noise; its mean and hyperparameters start the EIV iterations)
    hsgp_force_noise_rel: 0.0       # residual likelihoods: per-step sigma_f,t = value x |R_t| (0: off)
    hsgp_force_noise_cv: 0          # residual_exact: 1 = choose that value by cross-validation over the training steps
    hsgp_cv_folds: 2                # interleaved folds of the training steps for the cross-validation
    hsgp_cv_tol: 0.0                # > 0: smallest s within this many nats of the best CV score (plateau rule)
    hsgp_reaction_weight_rule: fixed  # fixed (hsgp_reaction_weight) | euclid (N_free / N_BC, EUCLID balancing)
    hsgp_force_noise_from_reaction: 0 # residual likelihoods: 1 = sigma_f,t,d = (N_BC / N_free) sigma_R,t,d
    (recipe, not hsgp_*) reaction_zero_components: [1]  # reactions known to vanish (R_y in a tensile test): fitted to 0
                                    # with sigma = reaction_noise_rel x |measured force| (>= reaction_noise_floor)
    hsgp_force_noise_evidence: 0    # residual_exact: 1 = choose that value by the evidence of the reaction-weighted
                                    # likelihood (set hsgp_reaction_weight >= 10, e.g. 10)
    hsgp_force_noise_joint: 0       # residual_exact: 1 = that value (log s) optimised with the kernel hyperparameters
                                    # on the reaction-weighted evidence by gradient (one solve; weight >= 10)
    hsgp_force_noise_joint_refit: 0 # with _joint: the weighted evidence only selects s; s fixed, hyperparameters and
                                    # posterior recomputed with reaction weight 1 (calibrated bands)
    hsgp_force_noise: 0.0           # residual likelihoods: free-node force-residual noise std (force units);
                                    # 0 = from disp_noise, < 0 = inferred by EM (residual_exact)
    hsgp_prior_mean: none           # none | linear_elastic | linear_elastic_fixed (mean frozen at the linear-elastic fit)
    hsgp_envelope: 1                # amplitude envelope s_dev, s_vol from the linear-elastic stage
    hsgp_warp: 1                    # GP inputs log(I_bar - 2), log J
    hsgp_continuation: 1            # load-step continuation over the training steps
    hsgp_num_basis_dev: 24          # eigenfunctions per deviatoric dimension
    hsgp_num_basis_vol: 32
    hsgp_box_factor: 8.0
    hsgp_max_outer: 60              # GP-stage iteration cap per continuation stage
    hsgp_amplitude_prior_scale: 1.0 # log-normal hyperprior widths (log units)
    hsgp_lengthscale_prior_scale: 1.0
    hsgp_reaction_weight: 1.0       # residual likelihoods only (residual, residual_exact, residual_learned): reaction
                                    # log-likelihood x weight (= sigma_R / sqrt(weight)); the EIV likelihood is unweighted
    hsgp_sigma_cap: 10.0            # GP amplitudes capped at this multiple of their prior centre; <= 0: no cap
    hsgp_stability_search: 1        # Gauss-Newton updates must keep the mean tangent positive definite
    hsgp_hyper_restarts: 1          # multi-start hyperparameter optimisation (prior centre, lengthscales x0.5, x2)
    hsgp_eiv_damping: 0.0           # initial relative damping of the EIV correction (K^2 + mu^2)^-1 K f; 0 = Newton K^-1 f
    hsgp_eiv_damping_decay: 0.1     # damping factor per GP iteration within a stage (annealed to exactly 0)
    hsgp_adaptive: 0                # 1: adaptive continuation with rollback (coarse full range first, bisect on failure)
    hsgp_hyper_iters_per_stage: 0   # >0: optimise hyperparameters only in the first k GP iterations of each stage
    hsgp_ls_floor_fraction: 0.0     # >0: soft lengthscale floor = fraction x span of the stage's data (GP input space)
    hsgp_tol: 1e-5                  # convergence tolerance (relative force change) of the final stage
    hsgp_tol_intermediate: 0.0      # >0: convergence tolerance of intermediate continuation stages (default: hsgp_tol)
    hsgp_hyper_delay: 0             # >0: hyperparameters fixed in the first k GP iterations of each later stage
    hsgp_cap_accept_tol: 0.0        # >0 (adaptive): a stage at the iteration cap with force change below this counts as converged
    hsgp_continuation_stages:       # optional explicit ladder of cumulative load-step sets (recipe only), e.g.
      - [1, 2, 3]                   #   default: [first two training steps], then one more step per stage
      - [1, 2, 3, 5]
      - [1, 2, 3, 5, 7]
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import yaml

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import matplotlib.pyplot as plt

from core.datasetclass import DatasetFactory
from archive.eiv.hsgp_inference import fit_hsgp, eiv_reaction_prediction
from core.hsgp_model import HSGPHyperelasticity
from core.loss_function import ellipticity_penalty
from core.material_models import get_material
from core.utils import load_f3x3_from_dataset
from plots.theme import apply_style, save_figure

HSGP_DEFAULTS = dict(hsgp_likelihood="eiv", hsgp_prior_mean="none", hsgp_envelope=1, hsgp_warp=1, hsgp_continuation=1,
                     hsgp_num_basis_dev=24, hsgp_num_basis_vol=32, hsgp_box_factor=8.0, hsgp_max_outer=60,
                     hsgp_amplitude_prior_scale=1.0, hsgp_lengthscale_prior_scale=1.0, hsgp_reaction_weight=1.0,
                     hsgp_sigma_cap=10.0, hsgp_stability_search=1, hsgp_hyper_restarts=1,
                     hsgp_eiv_damping=0.0, hsgp_eiv_damping_decay=0.1, hsgp_adaptive=0, hsgp_hyper_iters_per_stage=0,
                     hsgp_ls_floor_fraction=0.0, hsgp_tol_intermediate=0.0, hsgp_hyper_delay=0, hsgp_cap_accept_tol=0.0, hsgp_tol=1e-5,
                     hsgp_force_noise=0.0, hsgp_nigp_warmup="residual", hsgp_eiv_warmup="none",
                     hsgp_force_noise_rel=0.0, hsgp_force_noise_cv=0, hsgp_cv_folds=2, hsgp_force_noise_evidence=0,
                     hsgp_reaction_weight_rule="fixed", hsgp_force_noise_from_reaction=0, hsgp_cv_tol=0.0,
                     hsgp_force_noise_joint=0, hsgp_force_noise_joint_refit=0)


def parse_args():
    p = argparse.ArgumentParser(description="HSGP strain-energy extraction (one seed).")
    p.add_argument("--recipe", required=True, help="Recipe / experiment config.yaml")
    p.add_argument("--dataset_path", required=True, help="Dataset spec 'clean.npz?seed=..&disp_noise=..&load_noise=..' or file")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--batch_dir", required=True, help="Output directory (<experiment>/<seed>/extracted)")
    p.add_argument("--train_load_steps_indices", type=int, nargs="+", default=None)
    p.add_argument("--val_load_steps_indices", type=int, nargs="+", default=None)
    p.add_argument("--test_load_steps_indices", type=int, nargs="*", default=None)
    p.add_argument("--phase", choices=["fit", "plot", "all"], default="all")
    for k, v in HSGP_DEFAULTS.items():   # --hsgp_warp 0 etc. override the recipe
        p.add_argument(f"--{k}", type=type(v), default=None)
    return p.parse_args()


def external_work_density(prep, node_type, last_step):
    """Data energy scale: external work up to last_step / specimen area (as in extraction/train_unsupervised.py)."""
    u = np.asarray(prep["u_obs"] if "u_obs" in prep else prep["u"])[: last_step + 1]
    area = float(np.sum(np.asarray(prep["dA"])))
    R = np.asarray(prep["reaction_forces"])[: last_step + 1]
    nt = np.asarray(node_type)
    ux = u[:, nt[:, 3] == 1, 0].mean(axis=1) if np.any(nt[:, 3] == 1) else np.zeros(len(u))
    uy = u[:, nt[:, 4] == 1, 1].mean(axis=1) if np.any(nt[:, 4] == 1) else np.zeros(len(u))
    # the reaction side may move too (e.g. a DIC region of interest cut at both ends): use the relative displacement
    if np.any(nt[:, 1] == 1):
        ux = ux - u[:, nt[:, 1] == 1, 0].mean(axis=1)
    if np.any(nt[:, 2] == 1):
        uy = uy - u[:, nt[:, 2] == 1, 1].mean(axis=1)
    dW = 0.5 * (R[1:, 0] + R[:-1, 0]) * np.diff(ux) + 0.5 * (R[1:, 1] + R[:-1, 1]) * np.diff(uy)
    e = float(np.sum(dW)) / area
    return e if np.isfinite(e) and e > 0 else 1.0


def plot_reactions(model, F_steps, prep, train_steps, save_path, has_truth, sigma_R=None):
    """Reaction forces per load step: epistemic band vs true reactions (synthetic), predictive band vs measured.
    sigma_R: (n_steps, 2) reaction noise of the likelihood (default: the dataset's load_noise_std_steps)."""
    corr = model.info.get("likelihood", "eiv") not in ("residual", "residual_learned", "residual_exact")
    mean, cov = eiv_reaction_prediction(model, F_steps, prep["cells"], prep["node_type"], prep["dNdX"], prep["dA"], correction=corr)
    sd_epi = np.sqrt(np.stack([cov[:, 0, 0], cov[:, 1, 1]], axis=1))
    sig_R = np.maximum(np.asarray(prep["load_noise_std_steps"] if sigma_R is None else sigma_R, dtype=float), 1e-3)
    unmeasured = np.all(sig_R > 1e6, axis=0)                 # components without a load cell: epistemic band only
    sig_R = np.where(unmeasured[None, :], 0.0, sig_R)
    sd_pred = np.sqrt(sd_epi ** 2 + sig_R ** 2)
    R_obs = np.asarray(prep["reaction_forces"])
    panels = [("predictive", sd_pred, R_obs, "measured reaction")]
    if has_truth:
        panels.insert(0, ("epistemic", sd_epi, np.asarray(prep["reaction_forces_true"]), "true reaction"))
    steps = np.arange(F_steps.shape[0])
    is_tr = np.isin(steps, train_steps)
    apply_style()
    fig, axes = plt.subplots(2, len(panels), figsize=(7.0 * len(panels), 10.0), sharex=True, squeeze=False)
    out = {}
    for i, comp in enumerate(["x", "y"]):
        for j, (kind, sd, ref, lab) in enumerate(panels):
            ax = axes[i, j]
            m, s, r = mean[:, i], sd[:, i], ref[:, i]
            hit = np.abs(r - m) <= 1.96 * s
            ax.fill_between(steps, m - 1.96 * s, m + 1.96 * s, color="tab:blue", alpha=0.25, label="95% band")
            ax.plot(steps, m, color="tab:blue", lw=2, label="posterior mean")
            ax.plot(steps[~is_tr], r[~is_tr], "o", color="k", ms=5, label=f"{lab} (validation step)")
            ax.plot(steps[is_tr], r[is_tr], "s", color="tab:red", ms=7, label=f"{lab} (training step)")
            for k in np.where(~hit)[0]:
                ax.plot(steps[k], r[k], "x", color="tab:orange", ms=11, mew=2)
            r2 = 1 - np.sum((r - m) ** 2) / np.sum((r - r.mean()) ** 2)
            cov_all = hit.mean() * 100
            cov_val = hit[~is_tr].mean() * 100 if np.any(~is_tr) else float("nan")
            if kind == "predictive" and unmeasured[i]:
                ax.set_title(f"$R_{comp}$: not measured (epistemic band only)")
            else:
                ax.set_title(f"$R_{comp}$ {kind}: coverage {cov_all:.0f}% (val {cov_val:.0f}%), $R^2$ {r2:.4f}")
            ax.set_ylabel(f"$R_{comp}$")
            if i == 1:
                ax.set_xlabel("load step")
            if i == 0 and j == 0:
                ax.legend(fontsize=9)
            out[f"{kind}_{comp}"] = dict(coverage=float(cov_all), coverage_val=float(cov_val), r2=float(r2),
                                         z=((r - m) / s).tolist())
    fig.suptitle("Reaction forces (" + ("EIV-corrected prediction" if corr else "residual model, no state correction")
                 + " at the observed displacements)")
    save_figure(fig, os.path.join(save_path, "reaction_force_uncertainty.pdf"))
    return out


def plot_convergence(history, save_path):
    apply_style()
    it = [h["iteration"] for h in history]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].plot(it, [h["log_evidence"] for h in history], "o-"); ax[0].set_title("log evidence"); ax[0].set_xlabel("outer iteration")
    ax[1].semilogy(it, [h["force_change"] for h in history], "o-"); ax[1].set_title("relative change of internal forces")
    ax[2].semilogy(it, [h["sigma_u_x"] for h in history], "o-", label=r"$\sigma_{u,x}$")
    ax[2].semilogy(it, [h["sigma_u_y"] for h in history], "s-", label=r"$\sigma_{u,y}$"); ax[2].legend(); ax[2].set_title("displacement noise")
    stage_starts = [h["iteration"] for k, h in enumerate(history)
                    if k > 0 and h.get("continuation_stage", 0) != history[k - 1].get("continuation_stage", 0)]
    for a in ax:
        for x in stage_starts:
            a.axvline(x - 0.5, color="grey", ls=":")
    save_figure(fig, os.path.join(save_path, "hsgp_convergence.pdf"))


def main():
    args = parse_args()
    rec = yaml.safe_load(open(args.recipe)) or {}
    opt = {k: (getattr(args, k) if getattr(args, k) is not None else type(v)(rec.get(k, v))) for k, v in HSGP_DEFAULTS.items()}
    train_steps = args.train_load_steps_indices or rec.get("train_load_steps_indices")
    val_steps = args.val_load_steps_indices or rec.get("val_load_steps_indices", [])
    test_steps = args.test_load_steps_indices if args.test_load_steps_indices is not None else rec.get("test_load_steps_indices", [])
    save_path = args.batch_dir
    os.makedirs(save_path, exist_ok=True)

    prep = DatasetFactory.create("dataset/precomputed_vfm", data_path=args.dataset_path).get_data()
    has_truth = bool(rec.get("has_ground_truth", True)) and "reaction_forces_true" in prep
    mat_p = rec.get("material_params", {}) or {}
    # true material from the recipe (any family: dev/vol params, Ogden mu/alpha, ...)
    from core.material_models import get_material_from_config
    mat = get_material_from_config(dict(rec, material_model_name=rec.get("material_model_name", "isihara"))) if has_truth else None
    F_all = load_f3x3_from_dataset(prep, material_model=mat)                       # (n_steps, C, 3, 3) observed
    # incompressible plane stress (recipe or dataset): F33 = 1/det F_2D inside the model, no volumetric part
    ps_mode = str(rec.get("plane_stress_mode", prep.get("plane_stress_mode", "")) or "")
    incompressible_ps = str(prep.get("stress_mode", rec.get("stress_mode", ""))) == "plane_stress" and ps_mode == "incompressible"
    # reaction components with a load cell (e.g. [0] for a uniaxial machine): the others get no weight in the likelihood
    sigma_R_steps = np.array(prep["load_noise_std_steps"], dtype=float)
    if rec.get("reaction_components") is not None:
        unmeasured = [i for i in range(sigma_R_steps.shape[1]) if i not in [int(c) for c in rec["reaction_components"]]]
        sigma_R_steps[:, unmeasured] = 1e12
    if rec.get("reaction_noise_rel") is not None:      # load-cell noise relative to the reading (+ optional floor)
        R_abs = np.abs(np.asarray(prep["reaction_forces"], dtype=float))
        rel = np.maximum(float(rec["reaction_noise_rel"]) * R_abs, float(rec.get("reaction_noise_floor", 0.0)))
        meas = [int(c) for c in rec.get("reaction_components", range(sigma_R_steps.shape[1]))]
        sigma_R_steps[:, meas] = rel[:, meas]
    zero_comps = [int(c) for c in rec.get("reaction_zero_components", [])]
    if zero_comps:     # components known to vanish (e.g. R_y of a tensile test): observed 0, noise from the measured force
        rf = np.array(prep["reaction_forces"], dtype=float)
        rf[:, zero_comps] = 0.0
        prep["reaction_forces"] = rf
        meas = [int(c) for c in rec.get("reaction_components", range(rf.shape[1])) if int(c) not in zero_comps]
        scale = np.linalg.norm(rf[:, meas], axis=1)
        rel_z = float(rec.get("reaction_noise_rel", 0.01))
        sigma_R_steps[:, zero_comps] = np.maximum(rel_z * scale, float(rec.get("reaction_noise_floor", 0.0)))[:, None]
        print(f"[HSGP] reaction components {zero_comps} fitted to 0 with sigma = {rel_z:g} x |measured force| "
              f"(floor {float(rec.get('reaction_noise_floor', 0.0)):g})", flush=True)
    energy_scale = external_work_density(prep, prep["node_type"], max(train_steps))
    lin_prior_scale = float(rec.get("linear_elastic_prior_scale", 100.0 * energy_scale))
    disp_noise = float(rec.get("disp_noise", prep.get("disp_noise", 0.0)))
    post_path = os.path.join(save_path, "hsgp_posterior.npz")
    print(f"[HSGP] seed {args.seed} | steps {train_steps} | energy scale {energy_scale:.4g} | options {opt}", flush=True)

    if args.phase in ("fit", "all"):
        t0 = time.time()
        model, hist = fit_hsgp(
            F_all[jnp.array(train_steps)], np.asarray(prep["reaction_forces"])[train_steps],
            sigma_R_steps[train_steps], prep["cells"], prep["node_type"], prep["dNdX"], prep["dA"],
            energy_scale=energy_scale, lin_prior_scale=lin_prior_scale,
            num_basis_dev=opt["hsgp_num_basis_dev"], num_basis_vol=opt["hsgp_num_basis_vol"], box_factor=opt["hsgp_box_factor"],
            amplitude_prior_scale=opt["hsgp_amplitude_prior_scale"], lengthscale_prior_scale=opt["hsgp_lengthscale_prior_scale"],
            prior_mean=opt["hsgp_prior_mean"], envelope=bool(opt["hsgp_envelope"]), likelihood=opt["hsgp_likelihood"],
            sigma_u_known=None if float(opt["hsgp_force_noise"]) < 0 else disp_noise,    # < 0: sigma_f inferred (EM)
            reaction_weight=opt["hsgp_reaction_weight"], reaction_weight_rule=str(opt["hsgp_reaction_weight_rule"]),
            warp=bool(opt["hsgp_warp"]),
            continuation=bool(opt["hsgp_continuation"]), max_outer=opt["hsgp_max_outer"], step_labels=train_steps,
            sigma_cap=opt["hsgp_sigma_cap"], stability_search=bool(opt["hsgp_stability_search"]),
            hyper_restarts=bool(opt["hsgp_hyper_restarts"]), continuation_stages=rec.get("hsgp_continuation_stages"),
            eiv_damping=float(opt["hsgp_eiv_damping"]), eiv_damping_decay=float(opt["hsgp_eiv_damping_decay"]),
            adaptive=bool(opt["hsgp_adaptive"]), hyper_iters_per_stage=int(opt["hsgp_hyper_iters_per_stage"]),
            ls_floor_fraction=float(opt["hsgp_ls_floor_fraction"]),
            tol_intermediate=float(opt["hsgp_tol_intermediate"]) or None, hyper_delay=int(opt["hsgp_hyper_delay"]),
            cap_accept_tol=float(opt["hsgp_cap_accept_tol"]), incompressible_plane_stress=incompressible_ps,
            tol=float(opt["hsgp_tol"]),
            sigma_f_known=float(opt["hsgp_force_noise"]) if float(opt["hsgp_force_noise"]) > 0 else None,
            nigp_warmup=str(opt["hsgp_nigp_warmup"]), eiv_warmup=str(opt["hsgp_eiv_warmup"]),
            sigma_f_rel=float(opt["hsgp_force_noise_rel"]) or None, sigma_f_cv=bool(int(opt["hsgp_force_noise_cv"])),
            cv_folds=int(opt["hsgp_cv_folds"]), sigma_f_evidence=bool(int(opt["hsgp_force_noise_evidence"])),
            sigma_f_from_reaction=bool(int(opt["hsgp_force_noise_from_reaction"])), cv_tol=float(opt["hsgp_cv_tol"]),
            sigma_f_joint=bool(int(opt["hsgp_force_noise_joint"])),
            joint_refit_unweighted=bool(int(opt["hsgp_force_noise_joint_refit"])))
        fit_time = time.time() - t0
        model.save(post_path)
        # adaptive continuation may stop early: the posterior is then conditioned on fewer steps, and the teacher must
        # only be exported at those states (distillation reads train_load_steps_indices from config.json)
        valid_steps = model.info.get("valid_load_steps") or train_steps
        if model.info.get("truncated"):
            print(f"[HSGP] WARNING: posterior conditioned on load steps {valid_steps} only "
                  f"({model.info.get('adaptive_stop_reason')}; skipped {model.info.get('skipped_load_steps')})", flush=True)
        feats = jax.vmap(model.feature_extractor.extract)(F_all[jnp.array(valid_steps)].reshape(-1, 3, 3))
        I_obs = np.concatenate([np.asarray(feats[0]), np.asarray(feats[1])], axis=-1)
        np.save(os.path.join(save_path, "I_obs_all.npy"), I_obs)
        cfg_out = dict(rec, extraction_method="hsgp", dataset_path=args.dataset_path, seed=args.seed, batch_dir=save_path,
                       train_load_steps_indices=valid_steps, train_load_steps_requested=train_steps,
                       val_load_steps_indices=val_steps, test_load_steps_indices=test_steps,
                       data_energy_scale=energy_scale, hsgp_continuation_stages=rec.get("hsgp_continuation_stages"), linear_elastic_prior_scale=lin_prior_scale, hsgp_options=opt,
                       hsgp=dict(model.info, hyper={k: np.asarray(v).tolist() for k, v in model.hyper.items()}))
        json.dump(cfg_out, open(os.path.join(save_path, "config.json"), "w"), indent=4, default=float)
        json.dump(hist, open(os.path.join(save_path, "hsgp_history.json"), "w"), indent=2)
        print(f"[seed {args.seed}] fit done in {fit_time:.0f}s, converged {model.info['converged']}", flush=True)
        if args.phase == "fit":
            return
    else:
        model = HSGPHyperelasticity.load(post_path)
        hist = json.load(open(os.path.join(save_path, "hsgp_history.json")))
        I_obs = np.load(os.path.join(save_path, "I_obs_all.npy"))
        fit_time = model.info.get("fit_time", float("nan"))

    t0 = time.time()
    plot_convergence(hist, save_path)
    reac = plot_reactions(model, F_all, prep, train_steps, save_path, has_truth, sigma_R=sigma_R_steps)
    _, frac = ellipticity_penalty(model.psi_det, F_all[jnp.array(train_steps)], 8)
    metrics = dict(seed=args.seed, extraction_method="hsgp", extraction_time=fit_time, train_steps=train_steps,
                   val_steps=val_steps, log_evidence=model.info["log_evidence"], converged=model.info["converged"],
                   outer_iterations=model.info["outer_iterations"],
                   all_stages_converged=model.info.get("all_stages_converged"), stage_status=model.info.get("stage_status"),
                   sigma_u_x=model.noise["sigma_u_x"], sigma_u_y=model.noise["sigma_u_y"],
                   linear_elastic_mean=model.linear_elastic_summary(), small_strain_moduli=model.small_strain_moduli(),
                   fraction_unstable=float(frac), reaction_forces=reac,
                   force_noise=dict(sigma_f_rel=model.info.get("sigma_f_rel"), cv=model.info.get("force_noise_cv")),
                   hyper={k: np.asarray(v).tolist() for k, v in model.hyper.items()})
    if has_truth:   # synthetic benchmark: energy and stress against the true material
        from plots.training import (plot_training_r2, plot_training_r2_piola, plot_combined_validation,
                                    plot_energy_decomposition_validation)
        import gc

        def release():   # compiled functions keep their (large, closed-over) sample arrays alive: free between plots
            jax.clear_caches()
            gc.collect()
        release()   # the stability check and moduli above leave their compiled functions behind
        r2_res = plot_training_r2(model, mat, F_all, save_path, train_steps=train_steps, val_steps=val_steps, test_steps=test_steps)
        release()
        plot_training_r2_piola(model, mat, F_all, save_path, train_steps=train_steps, val_steps=val_steps,
                               test_steps=test_steps, n_pws_samples=64)
        release()
        plot_combined_validation(model, mat, save_path, 0, I_obs=I_obs)
        release()
        plot_energy_decomposition_validation(model, mat, save_path, I_obs=I_obs)
        metrics.update(r2=r2_res[0], rmse=r2_res[1], ec=r2_res[2],
                       r2_train=r2_res.train_metrics.get("r2"), ec_train=r2_res.train_metrics.get("ec"),
                       r2_val=r2_res.val_metrics.get("r2"), rmse_val=r2_res.val_metrics.get("rmse"),
                       ec_val=r2_res.val_metrics.get("ec"))
    else:   # experimental data: the extracted energy along the six standard deformation modes, without a truth
        from plots.training import plot_energy_decomposition_validation
        plot_energy_decomposition_validation(model, None, save_path, I_obs=I_obs)
    # data coverage in the GP input space: interpolation region and where the standard modes leave it
    from plots.training import plot_invariant_space_modes
    metrics["mode_exit_gamma"] = plot_invariant_space_modes(model, save_path, I_obs)
    json.dump(metrics, open(os.path.join(save_path, "extraction_metrics.json"), "w"), indent=4, default=float)
    print(f"[seed {args.seed}] plots and metrics done in {time.time() - t0:.0f}s -> {save_path} | "
          f"R2 val {metrics.get('r2_val', float('nan')):.4f} EC val {metrics.get('ec_val', float('nan')):.1f} | "
          f"moduli {metrics['small_strain_moduli']}", flush=True)


if __name__ == "__main__":
    main()
