"""
Pipeline step 3: distil a parameter distribution of an interpretable candidate model (core/distillation_models.py:
GMR / Ogden deviatoric, volumetric and fibre libraries) from the GP teacher's function-space samples, with
uqmodeldisc's Wasserstein distillation of a normalizing flow, then prune terms by Sobol sensitivity and re-distil.
The GP export (pipeline/export_gp.py) is created on demand if it does not exist yet.

Usage: python3 pipeline/distill.py --saved_model_dir <seed>/extracted --component dev|vol|aniso --override_out_dir <dir>
"""
import sys
import os
import subprocess
import argparse
import numpy as np
from pathlib import Path
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'UQInModelDiscovery')))

import torch

import uqmodeldisc.parameterdistillation.distillation as uq_distill

# Monkey-patch torch.sqrt in uqmodeldisc.parameterdistillation.distillation to prevent NaN gradients when L2 norm is 0
_orig_torch_sqrt = uq_distill.torch.sqrt
def _safe_torch_sqrt(x):
    return _orig_torch_sqrt(torch.clamp(x, min=1e-12))
uq_distill.torch.sqrt = _safe_torch_sqrt

# Checkpointed parameter samples (--checkpoints): every training run of distill_parameter_distribution_from_gp creates its
# flow through create_parameter_distribution; the wrapped flow counts its draws (num_iters_lipschitz + 1 per Wasserstein
# iteration) and, after k completed iterations (first draw of iteration k + 1), stores extra samples drawn under a
# forked RNG, so the training trajectory is identical to a run without checkpoints. One record per training run.
CHECKPOINTS = dict(iters=set(), draws_per_iter=11, n_samples=5000, model=None, runs=[])
_orig_create_distribution = uq_distill.create_parameter_distribution


def _create_distribution_with_checkpoints(*a, **k):
    dist = _orig_create_distribution(*a, **k)
    if not CHECKPOINTS["iters"]:
        return dist
    record = dict(calls=0, samples={})
    CHECKPOINTS["runs"].append(record)
    forward = dist.forward

    def counting_forward(num_samples):
        c = record["calls"]
        record["calls"] += 1
        it_done, rem = divmod(c, CHECKPOINTS["draws_per_iter"])
        if rem == 0 and it_done in CHECKPOINTS["iters"]:
            devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
            with torch.no_grad(), torch.random.fork_rng(devices=devices):
                raw = forward(CHECKPOINTS["n_samples"])
            model = CHECKPOINTS["model"]
            pop = model._parameter_population_matrix
            full = torch.matmul(raw, pop.T.to(raw.dtype)) if raw.shape[1] < pop.shape[0] else raw
            record["samples"][it_done] = full.detach().cpu().numpy()
        return forward(num_samples)
    dist.forward = counting_forward
    return dist


uq_distill.create_parameter_distribution = _create_distribution_with_checkpoints

from uqmodeldisc.parameterdistillation.distillation import (
    distill_parameter_distribution_from_gp,
    save_normalizing_flow_parameter_distribution,
    load_normalizing_flow_parameter_distribution,
)
from uqmodeldisc.io import ProjectDirectory
from uqmodeldisc.testcases import (
    test_case_identifier_uniaxial_tension,
    test_case_identifier_equibiaxial_tension,
    test_case_identifier_pure_shear,
)

from core.distillation_models import (  # noqa: E402
    MockGP, EnergyOutputSelector, PyTorchGMRModel, PyTorchGMRDevModel, PyTorchGMRVolModel,
    PyTorchGMRAnisoCompModel, PyTorchGMRAnisoModel, N_OGDEN_TERMS, OGDEN_ALPHA_INIT, OGDEN_PARAM_NAMES,
    OGDEN_MU_NAMES, OGDEN_ALPHA_NAMES,
)
from plots.plot_distillation_loss import plot_loss_monitoring  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--saved_model_dir", type=str, required=True)
    parser.add_argument("--material_model", type=str, default="gmr", help="Material model to distill (gmr/gmr_log with log terms, gmr_nolog without log terms, gmr_aniso for anisotropic terms)")
    parser.add_argument("--n_iterations", type=int, default=5000)
    parser.add_argument("--load_distilled_dir", type=str, default=None, help="Path to existing distilled model directory to load flow parameters without re-running Stage 1")
    parser.add_argument("--do_sensitivity", action="store_true", default=True, help="Perform Sobol sensitivity analysis and re-distill with sensitive parameters")
    parser.add_argument("--no_sensitivity", dest="do_sensitivity", action="store_false", help="Skip sensitivity analysis")
    parser.add_argument("--sobol_threshold", type=float, default=1e-4, help="Total Sobol index threshold for selecting sensitive parameters")
    parser.add_argument("--gmr_order", type=int, default=3,
                        help="GMR library: polynomial terms C_ij (I1_bar-3)^i (I2_bar-3)^j with 1 <= i + j <= gmr_order; "
                             "terms of order > 3 follow E in the parameter vector (C40, C31, C22, ...), so the indices of "
                             "--dev_params_disabled up to 9 are unchanged (core/material_library.py)")
    parser.add_argument("--ogden_fixed_alpha_stage", type=int, default=1,
                        help="Ogden libraries: 1 (default) = stage 1 fits the coefficients at the fixed exponents "
                             "alpha0 = -4..4, the Sobol analysis selects the terms, then the kept terms are re-distilled "
                             "with mu and alpha together; 0 = mu and alpha jointly from the start")
    parser.add_argument("--include_log_term", type=int, default=0,
                        help="1: fit the GMR log term E ln(I2_bar / 3); default 0: E stays in the layout but is fixed "
                             "at 0 (also implied by --material_model gmr_log)")
    parser.add_argument("--sobol_samples_factor", type=int, default=1024, help="Saltelli sample factor for sensitivity analysis")
    parser.add_argument("--sample_mode", type=str, default="dataset_f", choices=["standard_interp", "dataset_f", "dataset_f_random", "dataset_f_stratified", "dataset_f_uniform_energy", "dataset_all"], help="Sample deformation inputs from extraction dataset directly (with standard FPS, stratified high-strain FPS, or uniform energy spectrum sampling) or standard modes")
    parser.add_argument("--num_points", type=int, default=192, help="Number of points for GP joint evaluation and distillation")
    parser.add_argument("--num_func_samples", type=int, default=512, help="Number of functional samples drawn from GP during parameter distillation")
    parser.add_argument("--max_gamma", type=float, default=1.0, help="Max deformation intensity gamma when sample_mode is standard")
    parser.add_argument("--distill_target", type=str, default="sef", choices=["sef", "sef_stress", "sef_cauchy", "sef_split"], help="Target mode: strain energy function (sef), joint SEF + Piola stress (sef_stress), joint SEF + Cauchy stress (sef_cauchy), or separate DEV and VOL energy (sef_split)")
    parser.add_argument("--component", type=str, default="total", choices=["total", "dev", "vol", "aniso"], help="Component to distill (used with sef_split)")
    
    parser.add_argument("--dev_params_disabled", type=int, nargs="*", default=[], help="Indices of deviatoric parameters to disable")
    parser.add_argument("--vol_params_disabled", type=int, nargs="*", default=[], help="Indices of volumetric parameters to disable")
    parser.add_argument("--aniso_params_disabled", type=int, nargs="*", default=[], help="Indices of anisotropic parameters to disable")

    parser.add_argument("--override_out_dir", type=str, default=None, help="Explicitly specify the output directory for distilled model logs and artifacts (overriding timestamp generation)")
    parser.add_argument("--load_existing_sensitivities", action="store_true", help="Skip Sobol resampling and directly load existing sensitivity CSVs from out_dir")
    parser.add_argument("--sampling_mode", type=str, default=None, choices=["cached_L", "mvn", "pathwise"], help="GP functional draw sampling mode: cached_L (fastest, precomputed Cholesky factor L), mvn (torch.distributions.MultivariateNormal), or pathwise")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for PyTorch and Numpy")
    parser.add_argument("--checkpoints", type=int, nargs="*", default=[],
                        help="Wasserstein iterations at which the final distillation stage also saves parameter samples "
                             "(<component>_flow_samples_it<k>.npy, same layout as <component>_flow_samples.npy)")
    args = parser.parse_args()

    # Resolve sampling mode: CLI argument > recipe config.yaml/json > default ('cached_L')
    sampling_mode = args.sampling_mode
    if sampling_mode is None:
        for cdir in [args.saved_model_dir, os.path.dirname(os.path.abspath(args.saved_model_dir))]:
            yaml_path = os.path.join(cdir, "config.yaml")
            if os.path.exists(yaml_path):
                try:
                    import yaml
                    with open(yaml_path, "r") as yf:
                        ycfg = yaml.safe_load(yf) or {}
                        if "distillation_sampling_mode" in ycfg:
                            sampling_mode = ycfg["distillation_sampling_mode"]
                            break
                except Exception:
                    pass
            cfg_path = os.path.join(cdir, "config.json")
            if sampling_mode is None and os.path.exists(cfg_path):
                try:
                    import json
                    with open(cfg_path, "r") as jf:
                        jcfg = json.load(jf) or {}
                        if "distillation_sampling_mode" in jcfg:
                            sampling_mode = jcfg["distillation_sampling_mode"]
                            break
                except Exception:
                    pass
        if sampling_mode is None:
            sampling_mode = "cached_L"

    print(f"[DISTILLATION] Sampling Mode: '{sampling_mode}'")
    
    prefix = f"{args.component}_" if args.distill_target == "sef_split" else ""
    def pfx(filename): return prefix + filename

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    torch.set_default_dtype(torch.float64)
    # stable per-component offset (Python's str hash() is randomised per process, which made every run unseeded)
    import zlib
    comp_seed_offset = (zlib.crc32(args.component.encode()) % 10000) if args.component else 0
    effective_seed = (args.seed + comp_seed_offset) & 0x7fffffff

    def set_all_seeds(s: int):
        torch.manual_seed(s)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(s)
        np.random.seed(s)

    set_all_seeds(effective_seed)

    if args.sample_mode == "dataset_all":
        export_subfolder = "pytorch_export_dataset_all"
    elif args.sample_mode.startswith("dataset_f"):
        export_subfolder = f"pytorch_export_{args.sample_mode}_n{args.num_points}"
    else:   # standard_interp
        export_subfolder = "pytorch_export_standard_interp"
    if args.distill_target in ["sef_stress", "sef_cauchy"]:
        export_subfolder += f"_{args.distill_target}"
        
    export_dir = os.path.join(args.saved_model_dir, export_subfolder)
    if args.distill_target == "sef_split":
        mean_file = f"mean_{args.component}.npy"
        cov_file  = f"cov_{args.component}.npy"
    else:
        mean_file = "mean_psi.npy"
        cov_file  = "cov_psi.npy"

    if not os.path.exists(export_dir) or not os.path.exists(os.path.join(export_dir, mean_file)):
        print(f"'{export_dir}' or '{mean_file}' not found. Exporting GP to PyTorch first (sample_mode: {args.sample_mode}, max_gamma: {args.max_gamma}, distill_target: {args.distill_target})...")
        subprocess.run(["python3", "pipeline/export_gp.py", "--saved_model_dir", args.saved_model_dir, "--sample_mode", args.sample_mode, "--num_points", str(args.num_points), "--max_gamma", str(args.max_gamma), "--distill_target", args.distill_target, "--export_subfolder", export_subfolder], check=True)

    # Health check: verify export verification report
    report_file = os.path.join(export_dir, "export_verification_report.json")
    if os.path.exists(report_file):
        import json
        with open(report_file, "r") as rf:
            rep = json.load(rf)
        if rep.get("status") != "PASSED" or not rep.get("overall_healthy", False):
            errs = rep.get("errors", ["Unknown matrix defect"])
            raise RuntimeError(
                f"\n❌ [CRITICAL ERROR] Export verification failed for directory:\n   {export_dir}\n"
                f"Errors reported: {errs}\n"
                f"Distillation halted because exported covariance matrices are not healthy / positive-definite."
            )
        comp_key = args.component if args.distill_target == "sef_split" else "psi"
        comp_stats = rep.get("components", {}).get(comp_key, {})
        min_e = comp_stats.get("min_eigenvalue", "N/A")
        c_num = comp_stats.get("condition_number", "N/A")
        min_e_str = f"{min_e:.3e}" if isinstance(min_e, (int, float)) else str(min_e)
        c_num_str = f"{c_num:.2e}" if isinstance(c_num, (int, float)) else str(c_num)
        print(f"✅ [EXPORT VERIFICATION] Verified clean: component '{comp_key}' (min_eig={min_e_str}, cond={c_num_str})")
    else:
        print(f"ℹ️ [EXPORT VERIFICATION] No export_verification_report.json found in {export_dir}. Running on-the-fly verification...")
        from pipeline.export_gp import verify_and_report_export
        comp_dict = {}
        if args.distill_target == "sef_split":
            for c in ["dev", "vol", "aniso"]:
                mf = os.path.join(export_dir, f"mean_{c}.npy")
                cf = os.path.join(export_dir, f"cov_{c}.npy")
                if os.path.exists(mf) and os.path.exists(cf):
                    comp_dict[c] = {"mean": np.load(mf), "cov": np.load(cf)}
        else:
            mf = os.path.join(export_dir, "mean_psi.npy")
            cf = os.path.join(export_dir, "cov_psi.npy")
            if os.path.exists(mf) and os.path.exists(cf):
                comp_dict["psi"] = {"mean": np.load(mf), "cov": np.load(cf)}
        f3_data = np.load(os.path.join(export_dir, "f3x3.npy"))
        verify_and_report_export(export_dir, comp_dict, f3_data, args.sample_mode, args.distill_target)

    mean_psi = torch.tensor(np.load(os.path.join(export_dir, mean_file)), dtype=torch.float64, device=device)
    cov_psi = torch.tensor(np.load(os.path.join(export_dir, cov_file)), dtype=torch.float64, device=device)
    cov_psi = (cov_psi + cov_psi.T) / 2.0

    L_file = f"L_{args.component}.npy" if args.distill_target == "sef_split" else "L_psi.npy"
    L_path = os.path.join(export_dir, L_file)
    L_tensor = None

    if sampling_mode == "cached_L":
        if os.path.exists(L_path):
            L_tensor = torch.tensor(np.load(L_path), dtype=torch.float64, device=device)
            print(f"✅ Loaded precomputed Cholesky factor L from {L_file} (shape: {list(L_tensor.shape)})")
        else:
            print(f"⚠️ Precomputed Cholesky factor {L_file} not found. Computing and caching...")
            eigs, V = torch.linalg.eigh(cov_psi)
            min_eig = eigs.min().item()
            if min_eig < 1e-8:
                eigs = torch.clamp(eigs, min=1e-8)
                cov_psi = V @ torch.diag(eigs) @ V.T
                cov_psi = (cov_psi + cov_psi.T) / 2.0
            try:
                L_tensor = torch.linalg.cholesky(cov_psi)
            except RuntimeError:
                print("Fallback: Adding trace jitter before Cholesky.")
                cov_psi += 1e-6 * torch.eye(cov_psi.shape[0], dtype=torch.float64, device=device)
                L_tensor = torch.linalg.cholesky(cov_psi)
            np.save(L_path, L_tensor.cpu().numpy())
            print(f"Saved computed Cholesky factor to {L_file}")
    else:
        # Standard positive-definiteness adjustment for mvn mode if needed
        eigs, V = torch.linalg.eigh(cov_psi)
        min_eig = eigs.min().item()
        if min_eig < 1e-8:
            print(f"Adjusting numerical eigenvalues (min eigen: {min_eig:.3e}) to ensure positive-definiteness...")
            eigs = torch.clamp(eigs, min=1e-8)
            cov_psi = V @ torch.diag(eigs) @ V.T
            cov_psi = (cov_psi + cov_psi.T) / 2.0
            try:
                torch.linalg.cholesky(cov_psi)
            except RuntimeError:
                cov_psi += 1e-6 * torch.eye(cov_psi.shape[0], dtype=torch.float64, device=device)
    f3x3 = torch.tensor(np.load(os.path.join(export_dir, "f3x3.npy")), dtype=torch.float64, device=device)

    num_points = f3x3.shape[0]
    
    # Map deformation modes to test case identifiers for sensitivity analysis output grouping
    test_cases = torch.zeros(num_points, dtype=torch.int64, device=device)
    if args.sample_mode.startswith("dataset_"):
        # Heterogeneous full-field data: treat all as a single experimental load case
        test_cases[:] = test_case_identifier_equibiaxial_tension
    else:
        chunk = max(1, num_points // 6)
        test_cases[0:chunk] = test_case_identifier_uniaxial_tension
        test_cases[chunk:2*chunk] = test_case_identifier_equibiaxial_tension
        test_cases[2*chunk:3*chunk] = test_case_identifier_pure_shear
        test_cases[3*chunk:4*chunk] = test_case_identifier_uniaxial_tension
        test_cases[4*chunk:5*chunk] = test_case_identifier_equibiaxial_tension
        test_cases[5*chunk:] = test_case_identifier_pure_shear

    # For Sobol sensitivity analysis, prevent SALib NaN variance division at zero-strain origins (F=I)
    # by assigning an identifier (2=Biaxial Tension) ignored by Treloar output 0 screening.
    identity_matrix = torch.eye(3, dtype=torch.float64, device=device).unsqueeze(0)
    is_zero_strain = (f3x3 - identity_matrix).abs().max(dim=1)[0].max(dim=1)[0] < 1e-6
    test_cases[is_zero_strain] = 2 # test_case_identifier_biaxial_tension

    from core.material_models import get_material_from_dir
    try:
        true_model = get_material_from_dir(args.saved_model_dir, jit_P=False)
    except Exception:
        true_model = None

    # fibre directions of the anisotropic library: those of the fibre GP (the directions the teacher was fitted with),
    # else the true material's. Materials name their two families (a0, a1) or (a1, a2).
    theta1, theta2, theta_src = np.pi / 4.0, -np.pi / 4.0, "default (+-45 deg)"
    ang = lambda v: float(np.arctan2(float(v[1]), float(v[0])))
    try:
        from core.teacher import load_hsgp_teacher
        _teacher = load_hsgp_teacher(args.saved_model_dir)
    except Exception:
        _teacher = None
    if _teacher is not None and getattr(_teacher, "is_anisotropic", False) and getattr(_teacher, "a0", None) is not None:
        theta1 = ang(_teacher.a0)
        a_second = getattr(_teacher, "a1", None)
        theta2 = ang(a_second) if a_second is not None else theta1 + np.pi / 2.0
        theta_src = "fibre GP"
    elif true_model is not None and getattr(true_model, "a1", None) is not None and getattr(true_model, "a2", None) is not None:
        theta1, theta2, theta_src = ang(true_model.a1), ang(true_model.a2), "true material (a1, a2)"
    elif true_model is not None and getattr(true_model, "a0", None) is not None and getattr(true_model, "a1", None) is not None:
        theta1, theta2, theta_src = ang(true_model.a0), ang(true_model.a1), "true material (a0, a1)"
    elif true_model is not None and getattr(true_model, "a0", None) is not None:
        theta1, theta_src = ang(true_model.a0), "true material (a0; second family default)"
    if args.component == "aniso" or args.material_model in ["gmr_aniso", "aniso_gmr"]:
        print(f"Fibre directions of the anisotropic library: {np.degrees(theta1):.2f} and {np.degrees(theta2):.2f} deg "
              f"({theta_src})")

    mock_gp = MockGP(mean_psi, cov_psi, device=device, sampling_mode=sampling_mode, L=L_tensor)
    include_log = args.material_model not in ["gmr_nolog", "gmr_no_log"]
    # deviatoric library: "gmr" polynomial (+log) terms, "ogden" three Ogden terms, "gmr_ogden" both
    from core.material_library import library_from_material_model, is_dev_only
    dev_library = library_from_material_model(args.material_model)
    if is_dev_only(args.material_model) and args.component == "vol":
        raise ValueError(f"--material_model {args.material_model} has no volumetric part (--component dev or aniso)")
    if dev_library != "gmr" and args.distill_target != "sef_split":
        raise ValueError(f"--material_model {args.material_model} requires --distill_target sef_split")
    if args.material_model in ["gmr_aniso", "aniso_gmr"] or args.component == "aniso":
        if args.component == "dev":
            model = PyTorchGMRDevModel(num_points, device=device, distill_target=args.distill_target, include_log_terms=include_log,
                                       gmr_order=args.gmr_order)
        elif args.component == "vol":
            model = PyTorchGMRVolModel(num_points, device=device, distill_target=args.distill_target)
        elif args.component == "aniso":
            model = PyTorchGMRAnisoCompModel(num_points, device=device, theta1=theta1, theta2=theta2, distill_target=args.distill_target)
        else:
            model = PyTorchGMRAnisoModel(num_points, device=device, theta1=theta1, theta2=theta2, distill_target=args.distill_target)
    else:
        if args.component == "dev":
            model = PyTorchGMRDevModel(num_points, device=device, distill_target=args.distill_target,
                                       include_log_terms=include_log, library=dev_library, gmr_order=args.gmr_order)
        elif args.component == "vol":
            model = PyTorchGMRVolModel(num_points, device=device, distill_target=args.distill_target)
        else:
            if args.gmr_order != 3:
                raise NotImplementedError("--gmr_order != 3 needs --distill_target sef_split (the combined dev + vol "
                                          "model has the third-order library only)")
            model = PyTorchGMRModel(num_points, device=device, distill_target=args.distill_target, include_log_terms=include_log)
    
    global_disabled = set()
    
    if args.distill_target == "sef_split":
        if args.component == "dev":
            global_disabled.update(args.dev_params_disabled)
        elif args.component == "vol":
            global_disabled.update(args.vol_params_disabled)
        elif args.component == "aniso":
            global_disabled.update(args.aniso_params_disabled)
    else:
        num_dev = 10 if include_log else 9
        num_vol = 3
        
        for idx in args.dev_params_disabled:
            global_disabled.add(idx)
        for idx in args.vol_params_disabled:
            global_disabled.add(num_dev + idx)
        for idx in args.aniso_params_disabled:
            global_disabled.add(num_dev + num_vol + idx)
            
    # GMR log term E ln(I2_bar / 3): kept as a column of the layout (fixed at 0) unless asked for
    keep_log = bool(args.include_log_term) or "_log" in str(args.material_model).lower()
    if not keep_log and "$E$" in model.parameter_names:
        global_disabled.add(list(model.parameter_names).index("$E$"))
        print("GMR log term E is not part of the library (fixed at 0; --include_log_term 1 or gmr_log to fit it)")

    full_param_names_master = list(model.parameter_names)
    if global_disabled:
        model.deactivate_parameters(list(global_disabled))
        model.reduce_to_activated_parameters()
        print(f"Disabled parameters {list(global_disabled)}. Reduced active parameters ({model.num_parameters}): {model.parameter_names}")

    # Ogden library, staged: stage 1 fits the coefficients (mu_k, and polynomial terms) at the fixed start exponents
    # alpha0 = OGDEN_ALPHA_INIT (the alpha columns are inactive, i.e. a_k = 0 -> alpha_k = alpha0_k); the Sobol analysis
    # then selects the terms; the final stage re-distils the kept terms with mu_k and alpha_k together.
    ogden_staged = (getattr(model, "use_ogden", False) and bool(args.ogden_fixed_alpha_stage)
                    and not (args.load_distilled_dir and os.path.exists(args.load_distilled_dir)))
    base_state = None
    if ogden_staged:
        base_state = (model._parameter_population_matrix.clone(), tuple(model.parameter_names),
                      model.parameter_scales.clone())
        model.reduce_model_to_parameter_names([n for n in model.parameter_names if n not in OGDEN_ALPHA_NAMES])
        print(f"Ogden stage 1: exponents fixed at {OGDEN_ALPHA_INIT}; fitting ({model.num_parameters}): {model.parameter_names}")

    def expand_to_joint(selected_names):
        """Back to the full (post-disable) parameter set, keeping the selected names and the exponent of every kept mu_k."""
        pop, names, scales = base_state
        model._parameter_population_matrix = pop.clone()
        model.parameter_names = names
        model.num_parameters = len(names)
        model.parameter_scales = scales.clone()
        model.reset_parameter_deactivations()
        sel = set(selected_names)
        keep = [n for n in names if n in sel or (n in OGDEN_ALPHA_NAMES and OGDEN_MU_NAMES[OGDEN_ALPHA_NAMES.index(n)] in sel)]
        model.reduce_model_to_parameter_names(keep)
        print(f"Ogden joint stage: mu and alpha of the kept terms together ({model.num_parameters}): {model.parameter_names}")

    output_selector = EnergyOutputSelector(num_outputs=mean_psi.shape[0])
    CHECKPOINTS.update(iters={int(k) for k in args.checkpoints if 0 < int(k) < args.n_iterations}, model=model)

    from uqmodeldisc.settings import Settings
    settings = Settings()
    
    if args.load_distilled_dir and os.path.exists(args.load_distilled_dir):
        out_dir = os.path.abspath(args.load_distilled_dir)
        log_mode = "a"
    else:
        from core.material_library import gmr_names, gmr_order_from_length
        n_true = len(getattr(true_model, "dev_params", None) or [])
        dev_names = [("$E$" if n == "E" else f"$C_{{{n[1:]}}}$")
                     for n in gmr_names(max(args.gmr_order, gmr_order_from_length(n_true) if n_true else 3))]
        vol_names = ["$D_{1}$", "$D_{2}$", "$D_{3}$"]
        aniso_names = ["$C_{42}$", "$C_{43}$", "$C_{44}$", "$C_{62}$", "$C_{63}$", "$C_{64}$"]
        
        true_params = {}
        if hasattr(true_model, "dev_params"):
            for name, val in zip(dev_names, true_model.dev_params):
                if abs(val) > 1e-12:
                    true_params[name] = float(val)
        if hasattr(true_model, "vol_params"):
            for name, val in zip(vol_names, true_model.vol_params):
                if abs(val) > 1e-12:
                    true_params[name] = float(val)
        if hasattr(true_model, "aniso_params"):
            for name, val in zip(aniso_names, true_model.aniso_params):
                if abs(val) > 1e-12:
                    true_params[name] = float(val)
        if hasattr(true_model, "mu_params") and hasattr(true_model, "alpha_params"):     # Ogden true material
            for k, (mu_k, a_k) in enumerate(zip(true_model.mu_params, true_model.alpha_params)):
                true_params[OGDEN_PARAM_NAMES[2 * k]] = float(mu_k)
                true_params[OGDEN_PARAM_NAMES[2 * k + 1]] = float(a_k)

        source_type = "exp" if any(x in args.saved_model_dir.lower() for x in ["exp", "dic", "18617429"]) else "syn"

        if args.override_out_dir:
            out_dir = os.path.abspath(args.override_out_dir)
        else:
            saved_dir_name = os.path.basename(os.path.normpath(args.saved_model_dir))
            out_dir = os.path.abspath(os.path.join("results", "distilled_models", saved_dir_name))
        log_mode = "w"
    os.makedirs(out_dir, exist_ok=True)

    # Save distillation configuration
    import json
    import yaml
    config_dict = vars(args)
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(config_dict, f, indent=4)
    with open(os.path.join(out_dir, "config.yaml"), "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False)
    if args.distill_target == "sef_split":
        with open(os.path.join(out_dir, pfx("config.json")), "w") as f:
            json.dump(config_dict, f, indent=4)
        with open(os.path.join(out_dir, pfx("config.yaml")), "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False)
    
    with open(os.path.join(out_dir, pfx("source_extraction_dir.txt")), "w") as f:
        f.write(args.saved_model_dir)
    
    class TeeLogger(object):
        def __init__(self, filename, mode="w"):
            self.terminal = sys.stdout
            self.log = open(filename, mode, encoding="utf-8")
        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)
            self.log.flush()
        def flush(self):
            self.terminal.flush()
            self.log.flush()
            
    sys.stdout = TeeLogger(os.path.join(out_dir, pfx("distillation_log.txt")), mode=log_mode)
    print(f"\nLogging distillation outputs to {os.path.join(out_dir, pfx('distillation_log.txt'))}")
    
    settings.PROJECT_DIR = Path(out_dir)
    project_dir = ProjectDirectory(settings)
    time_records = {}
    pipeline_start_time = time.time()

    if args.load_distilled_dir and os.path.exists(os.path.join(out_dir, "output", "normalizing_flow_parameters")):
        print("\n========================================================================")
        model_type_str = f"({args.component.upper()} Split Model)" if args.distill_target == "sef_split" else "(Full Model)"
        print(f"=== Stage 1: Loading Existing Distilled Model (Skipping Training) {model_type_str} ===")
        print("========================================================================\n")
        distribution = load_normalizing_flow_parameter_distribution(
            model=model,
            output_subdirectory=args.component,
            project_directory=project_dir,
            device=device
        )
    else:
        print("\n========================================================================")
        model_type_str = f"({args.component.upper()} Split Model)" if args.distill_target == "sef_split" else "(Full Model)"
        print(f"=== Stage 1: Initial Parameter Distillation {model_type_str} ===")
        print("========================================================================\n")
        t0_stage1 = time.time()
        set_all_seeds(effective_seed)

        distribution = distill_parameter_distribution_from_gp(
            gp=mock_gp,
            model=model,
            output_selector=output_selector,
            distribution_type="normalizing flow",
            inputs=f3x3,
            test_cases=test_cases,
            num_func_samples=args.num_func_samples,
            lipschitz_penalty_coefficient=10.0,
            num_iters_wasserstein=args.n_iterations,
            num_layers_lipschitz_nn=3,
            relative_width_lipschitz_nn=1.0,
            num_iters_lipschitz=10,
            output_subdirectory=args.component,
            project_directory=project_dir,
            device=device,
            stage_tag=pfx("before_sensitivity"),
        )
        t_stage1_dur = time.time() - t0_stage1
        time_records['Stage 1 (Initial Distillation)'] = t_stage1_dur
        print(f"\n[Time Usage] Stage 1 finished in {t_stage1_dur / 60:.2f} minutes ({t_stage1_dur:.2f} seconds).\n")
        
        save_normalizing_flow_parameter_distribution(
            distribution=distribution,
            output_subdirectory=args.component,
            project_directory=project_dir,
            device=device
        )

        plot_loss_monitoring(
            os.path.join(out_dir, f"loss_history_{pfx('before_sensitivity')}.npz"),
            title_suffix=f"(Stage 1: Before Sensitivity Analysis - {args.component.upper()})" if args.distill_target == "sef_split" else "(Stage 1: Before Sensitivity Analysis)",
            output_path=os.path.join(out_dir, pfx("loss_monitoring_before_sensitivity.pdf")),
            window_size=min(100, max(10, args.n_iterations // 50))
        )
        
        if args.n_iterations >= 100:
            last_10_percent = max(10, int(0.1 * args.n_iterations))
            plot_loss_monitoring(
                os.path.join(out_dir, f"loss_history_{pfx('before_sensitivity')}.npz"),
                title_suffix=f"(Stage 1: Before Sensitivity Analysis - Last 10% - {args.component.upper()})" if args.distill_target == "sef_split" else "(Stage 1: Before Sensitivity Analysis - Last 10%)",
                output_path=os.path.join(out_dir, pfx("loss_monitoring_before_sensitivity_last_10_percent.pdf")),
                window_size=min(100, max(10, last_10_percent // 50)),
                last_n_iterations=last_10_percent
            )

    # Save parameter distribution plot immediately before running sensitivity analysis
    try:
        print("\nSaving parameter distribution plot before sensitivity analysis...")
        with torch.no_grad():
            raw_samples_pre = distribution.sample(5000)
            if raw_samples_pre.shape[1] < len(full_param_names_master):
                full_samples_pre = torch.matmul(raw_samples_pre, model._parameter_population_matrix.T.to(raw_samples_pre.dtype))
                samples_pre = full_samples_pre.cpu().numpy()
            else:
                samples_pre = raw_samples_pre.cpu().numpy()
            if hasattr(model, "physical_samples"):
                samples_pre = model.physical_samples(samples_pre, full_param_names_master)
            
        full_param_names_pre = list(full_param_names_master)
        pre_samples_path = os.path.join(out_dir, pfx("flow_samples_before_sensitivity.npy"))
        np.save(pre_samples_path, samples_pre)
        print(f"Saved pre-sensitivity parameter samples to {pre_samples_path}")
        
        import matplotlib.pyplot as plt
        import seaborn as sns
        import pandas as pd
        df_pre = pd.DataFrame(samples_pre, columns=full_param_names_pre)
        cols_pre = 4
        rows_pre = (len(full_param_names_pre) + cols_pre - 1) // cols_pre
        fig, axes_pre = plt.subplots(rows_pre, cols_pre, figsize=(cols_pre*4, rows_pre*4))
        axes_pre = axes_pre.flatten()
        means_pre = df_pre.mean()
        for i, col in enumerate(full_param_names_pre):
            sns.histplot(df_pre[col], ax=axes_pre[i], color='#16a085', bins=30, kde=True, edgecolor="black", alpha=0.7)
            mean_val = means_pre[col]
            axes_pre[i].axvline(mean_val, color='#c0392b', linestyle='--', linewidth=2, label=f"Mean: {mean_val:.4f}")
            if col in true_params:
                true_val = true_params[col]
                axes_pre[i].axvline(true_val, color='#27ae60', linestyle=':', linewidth=2, label=f"True: {true_val:.4f}")
                axes_pre[i].set_title(f"{col} (Pre-Sens)\nMean: {mean_val:.4f} (True: {true_val:.4f})", color='#2c3e50', fontsize=11, fontweight='bold')
            else:
                axes_pre[i].set_title(f"{col} (Pre-Sens)\nMean: {mean_val:.4f}", color='#2c3e50', fontsize=11, fontweight='bold')
            axes_pre[i].legend(fontsize=9)
        for j in range(len(full_param_names_pre), len(axes_pre)):
            axes_pre[j].set_visible(False)
        plt.tight_layout()
        pre_plot_path = os.path.join(out_dir, pfx(f"distributions_before_sensitivity_{args.material_model}.pdf"))
        plt.savefig(pre_plot_path, dpi=200)
        plt.close()
        print(f"Saved pre-sensitivity distribution plot to {pre_plot_path}")
    except Exception as e:
        print(f"Error saving pre-sensitivity distribution plot: {e}")


    if args.do_sensitivity:
        print("\n========================================================================")
        print("=== Stage 2: Sobol Sensitivity Analysis & Model Selection ===")
        print("========================================================================\n")
        t0_stage2 = time.time()
        from uqmodeldisc.models.modelselection import perform_sobol_sensitivity_analysis
        sensitivities_dir = os.path.join(out_dir, "sensitivities")
        os.makedirs(sensitivities_dir, exist_ok=True)
        
        if args.load_existing_sensitivities:
            print("Loading existing Sobol sensitivity indices from output/sensitivities...")
            import pandas as pd
            sens_dir_name = f"{args.component}_sensitivities" if args.distill_target == "sef_split" else "sensitivities"
            csv_dir = os.path.join(out_dir, "output", sens_dir_name)
            if not os.path.exists(os.path.join(csv_dir, "total_sobol_indices_statistics_output_0.csv")):
                csv_dir = os.path.join(out_dir, sens_dir_name)
            dfs = [pd.read_csv(os.path.join(csv_dir, f"total_sobol_indices_statistics_output_{i}.csv")) for i in range(model._output_dim)]
            param_cols = [c for c in dfs[0].columns if c not in ["Unnamed: 0", ""]]
            means_list = np.array([df.iloc[0][param_cols].values.astype(float) for df in dfs])
            mean_total_indices_outputs = np.nan_to_num(np.nanmean(means_list, axis=0), nan=0.0)
            
            relevant_parameter_indices = np.where(mean_total_indices_outputs >= args.sobol_threshold)[0].tolist()
            model.deactivate_all_parameters()
            model.activate_parameters(relevant_parameter_indices)
            print(f"Loaded active parameters based on threshold {args.sobol_threshold}: {model.get_active_parameter_names()}")
        else:
            from uqmodeldisc.models.modelselection import perform_sobol_sensitivity_analysis
            perform_sobol_sensitivity_analysis(
                model=model,
                parameter_distribution=distribution,
                num_samples_factor=args.sobol_samples_factor,
                data_set_label="custom",
                inputs=f3x3,
                test_cases=test_cases,
                select_model=True,
                total_sobol_index_thresshold=args.sobol_threshold,
                output_subdirectory=f"{args.component}_sensitivities" if args.distill_target == "sef_split" else "sensitivities",
                project_directory=project_dir,
                device=device,
            )

        # Generate custom bar plot for Sobol indices
        try:
            import matplotlib.pyplot as plt
            import pandas as pd
            sens_dir_name = f"{args.component}_sensitivities" if args.distill_target == "sef_split" else "sensitivities"
            csv_dir = os.path.join(out_dir, "output", sens_dir_name)
            if not os.path.exists(os.path.join(csv_dir, "total_sobol_indices_statistics_output_0.csv")):
                csv_dir = os.path.join(out_dir, sens_dir_name)
            stats_path = os.path.join(csv_dir, "total_sobol_indices_statistics_output_0.csv")
            first_stats_path = os.path.join(csv_dir, "first_sobol_indices_statistics_output_0.csv")
            if os.path.exists(stats_path) and os.path.exists(first_stats_path):
                df_tot = pd.read_csv(stats_path).fillna(0.0)
                df_first = pd.read_csv(first_stats_path).fillna(0.0)
                
                param_cols = [c for c in df_tot.columns if c not in ["Unnamed: 0", ""]]
                
                # Filter to only active parameters for this component
                if hasattr(model, 'parameter_names') and model.parameter_names:
                    param_cols = [c for c in param_cols if c in model.parameter_names]
                
                tot_means = df_tot.iloc[0][param_cols].values.astype(float)
                first_means = df_first.iloc[0][param_cols].values.astype(float)
                
                raw_path = os.path.join(csv_dir, "total_sobol_indices_output_0.csv")
                df_raw = pd.read_csv(raw_path)
                tot_maxs = df_raw[param_cols].max().values.astype(float)
                
                sorted_indices = np.argsort(tot_means)[::-1]
                sorted_param_cols = [param_cols[i] for i in sorted_indices]
                sorted_tot_means = tot_means[sorted_indices]
                sorted_tot_maxs = tot_maxs[sorted_indices]
                
                denominator = max(np.sum(first_means), np.sum(tot_means))
                if denominator == 0:
                    denominator = 1.0
                    
                norm_tot_means = sorted_tot_means / denominator
                est_coverage_frac = np.cumsum(sorted_tot_means) / denominator
                est_coverage_pct = est_coverage_frac * 100.0
                
                x = np.arange(len(sorted_param_cols))
                width = 0.25
                
                fig, ax1 = plt.subplots(figsize=(14, 7))
                
                ax1.bar(x - width, sorted_tot_means, width, label="Mean Total-Order", color="#008080", alpha=0.9)
                ax1.bar(x, sorted_tot_maxs, width, label="Max Total-Order", color="#20B2AA", alpha=0.9)
                ax1.bar(x + width, norm_tot_means, width, label="Norm. Total-Order", color="#48D1CC", alpha=0.9)
                
                ax1.set_yscale('log')
                ax1.set_ylim(bottom=max(1e-5, args.sobol_threshold * 0.1))
                ax1.axhline(args.sobol_threshold, color='black', linestyle='--', linewidth=1.5, label=f"Threshold ({args.sobol_threshold})")
                
                ax1.set_ylabel('Sobol Sensitivity Index (Log Scale)', fontsize=12, fontweight='bold')
                ax1.set_title('Material Parameter Sensitivity (Sobol Indices)', fontsize=14, fontweight='bold')
                ax1.set_xticks(x)
                ax1.set_xticklabels(sorted_param_cols, fontsize=11, fontweight='bold')
                
                ax2 = ax1.twinx()
                ax2.plot(x, est_coverage_pct, color='black', marker='o', linestyle='-', linewidth=2, markersize=6, label="Estimated Coverage (EC)")
                ax2.set_ylabel('Estimated Coverage (%)', color='black', fontsize=12, fontweight='bold')
                ax2.set_ylim(0, 105)
                ax2.set_yticks([0, 20, 40, 60, 80, 95, 100])
                ax2.tick_params(axis='y', labelcolor='black')
                ax2.axhline(100, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
                ax2.axhline(95, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
                
                lines_1, labels_1 = ax1.get_legend_handles_labels()
                lines_2, labels_2 = ax2.get_legend_handles_labels()
                ax1.legend(lines_1 + lines_2, labels_1 + labels_2, fontsize=11, loc='center right', bbox_to_anchor=(1.0, 0.6))
                
                plt.tight_layout()
                sobol_plot_path = os.path.join(out_dir, pfx("sobol_sensitivity_indices.pdf"))
                plt.savefig(sobol_plot_path, dpi=200)
                plt.close()
                print(f"Saved Sobol sensitivity bar plot to {sobol_plot_path}")
        except Exception as e:
            print(f"Error plotting Sobol indices: {e}")

        t_stage2_dur = time.time() - t0_stage2
        time_records['Stage 2 (Sobol Sensitivity Analysis)'] = t_stage2_dur
        print(f"\n[Time Usage] Stage 2 finished in {t_stage2_dur / 60:.2f} minutes ({t_stage2_dur:.2f} seconds).\n")

        # Custom parameter selection logic
        try:
            import pandas as pd
            sens_dir_name = f"{args.component}_sensitivities" if args.distill_target == "sef_split" else "sensitivities"
            csv_dir = os.path.join(out_dir, sens_dir_name)
            stats_path = os.path.join(csv_dir, "total_sobol_indices_statistics_output_0.csv")
            first_stats_path = os.path.join(csv_dir, "first_sobol_indices_statistics_output_0.csv")
            
            if os.path.exists(stats_path) and os.path.exists(first_stats_path):
                df_tot = pd.read_csv(stats_path).fillna(0.0)
                df_first = pd.read_csv(first_stats_path).fillna(0.0)
                
                param_cols = [c for c in df_tot.columns if c not in ["Unnamed: 0", ""]]
                tot_means = df_tot.iloc[0][param_cols].values.astype(float)
                first_means = df_first.iloc[0][param_cols].values.astype(float)
                
                sorted_indices = np.argsort(tot_means)[::-1]
                
                selected_indices = [i for i, mean in enumerate(tot_means) if mean >= args.sobol_threshold]
                print(f"\n[Pruning Mode: Threshold] Selecting parameters with Total-Order > {args.sobol_threshold}.")
                model.deactivate_all_parameters()
                model.activate_parameters(selected_indices)
                
        except Exception as e:
            print(f"Error during custom parameter selection logic: {e}")
            print("Falling back to UQModelDisc default selection.")

        if getattr(model, "use_ogden", False):
            # an Ogden term is kept only as a pair: a pruned exponent (alpha = 0) would leave mu sum ln^2 lam_bar, and a
            # pruned mu leaves a free exponent without effect
            names_now, active_idx = list(model.parameter_names), set(model.get_active_parameter_indices())
            for k in range(1, N_OGDEN_TERMS + 1):
                mu_n, al_n = f"$\\mu_{{{k}}}$", f"$\\alpha_{{{k}}}$"
                if mu_n in names_now and al_n in names_now:
                    pair = [names_now.index(mu_n), names_now.index(al_n)]
                    if sum(i in active_idx for i in pair) == 1:
                        model.deactivate_parameters(pair)
                        print(f"Ogden term {k}: only one of (mu_{k}, alpha_{k}) is sensitive; pruning the whole term.")
        active_names = model.get_active_parameter_names()
        print(f"Selected sensitive parameters ({len(active_names)}): {active_names}")
        if len(active_names) == 0:
            print("Warning: No parameters exceeded Sobol threshold! Keeping all parameters active.")
            model.activate_parameters(list(range(model.num_parameters)))
            active_names = model.get_active_parameter_names()
        if len(active_names) < model.num_parameters or ogden_staged:
            print("\n========================================================================")
            model_type_str = f"({args.component.upper()} Split Model)" if args.distill_target == "sef_split" else "(Sparse Model)"
            print(f"=== Stage 2: Target Sub-Parameter Distillation {model_type_str} ===")
            print("========================================================================\n")
            if ogden_staged:
                expand_to_joint(active_names)
            else:
                model.reduce_to_activated_parameters()
            print(f"Reduced model active parameters ({model.num_parameters}): {model.parameter_names}")
            t0_stage3 = time.time()
            set_all_seeds(effective_seed)
            
            distribution = distill_parameter_distribution_from_gp(
                gp=mock_gp,
                model=model,
                output_selector=output_selector,
                distribution_type="normalizing flow",
                inputs=f3x3,
                test_cases=test_cases,
                num_func_samples=args.num_func_samples,
                lipschitz_penalty_coefficient=10.0,
                num_iters_wasserstein=args.n_iterations,
                num_layers_lipschitz_nn=3,
                relative_width_lipschitz_nn=1.0,
                num_iters_lipschitz=10,
                output_subdirectory=args.component,
                project_directory=project_dir,
                device=device,
                stage_tag=pfx("after_sensitivity"),
            )
            t_stage3_dur = time.time() - t0_stage3
            time_records['Stage 3 (Re-Distillation Pruned Model)'] = t_stage3_dur
            print(f"\n[Time Usage] Stage 3 finished in {t_stage3_dur / 60:.2f} minutes ({t_stage3_dur:.2f} seconds).\n")
            
            plot_loss_monitoring(
                os.path.join(out_dir, f"loss_history_{pfx('after_sensitivity')}.npz"),
                title_suffix=f"(Stage 2: After Sensitivity Analysis - {args.component.upper()})" if args.distill_target == "sef_split" else "(Stage 2: After Sensitivity Analysis)",
                output_path=os.path.join(out_dir, pfx("loss_monitoring_after_sensitivity.pdf")),
                window_size=min(100, max(10, args.n_iterations // 50))
            )
            
            if args.n_iterations >= 100:
                last_10_percent = max(10, int(0.1 * args.n_iterations))
                plot_loss_monitoring(
                    os.path.join(out_dir, f"loss_history_{pfx('after_sensitivity')}.npz"),
                    title_suffix=f"(Stage 2: After Sensitivity Analysis - Last 10% - {args.component.upper()})" if args.distill_target == "sef_split" else "(Stage 2: After Sensitivity Analysis - Last 10%)",
                    output_path=os.path.join(out_dir, pfx("loss_monitoring_after_sensitivity_last_10_percent.pdf")),
                    window_size=min(100, max(10, last_10_percent // 50)),
                    last_n_iterations=last_10_percent
                )
        else:
            print("All material parameters were determined to be sensitive! No model reduction needed.")

    if ogden_staged and not args.do_sensitivity:   # no term selection: all terms, mu and alpha together
        expand_to_joint(list(model.parameter_names))
        set_all_seeds(effective_seed)
        distribution = distill_parameter_distribution_from_gp(
            gp=mock_gp, model=model, output_selector=output_selector, distribution_type="normalizing flow",
            inputs=f3x3, test_cases=test_cases, num_func_samples=args.num_func_samples,
            lipschitz_penalty_coefficient=10.0, num_iters_wasserstein=args.n_iterations, num_layers_lipschitz_nn=3,
            relative_width_lipschitz_nn=1.0, num_iters_lipschitz=10, output_subdirectory=args.component,
            project_directory=project_dir, device=device, stage_tag=pfx("after_sensitivity"))

    # Save the PyTorch Normalizing Flow model weights
    save_normalizing_flow_parameter_distribution(
        distribution=distribution,
        output_subdirectory=args.component,
        project_directory=project_dir,
        device=device
    )

    # Draw 5000 samples and map back to full 12D space if reduced
    with torch.no_grad():
        samples = distribution.sample(5000)
        full_param_names = full_param_names_master
        if samples.shape[1] < len(full_param_names):
            full_samples = torch.matmul(samples, model._parameter_population_matrix.T.to(samples.dtype))
            samples_np = full_samples.cpu().numpy()
        else:
            samples_np = samples.cpu().numpy()
        if hasattr(model, "physical_samples"):
            samples_np = model.physical_samples(samples_np, full_param_names)

        samples_path = os.path.join(settings.PROJECT_DIR, pfx("flow_samples.npy"))
        np.save(samples_path, samples_np)
        print(f"Saved distilled PyTorch parameter samples (shape: {samples_np.shape}) to {samples_path}")
        if args.checkpoints:   # samples of the final training stage after k Wasserstein iterations (k >= n_iterations: final)
            run = next((r for r in reversed(CHECKPOINTS["runs"]) if r["calls"] > 0), None)
            for k in sorted(set(int(k) for k in args.checkpoints)):
                if k >= args.n_iterations:
                    arr = samples_np
                elif run is not None and k in run["samples"]:
                    arr = run["samples"][k]
                    if hasattr(model, "physical_samples"):
                        arr = model.physical_samples(arr, full_param_names)
                else:
                    print(f"Checkpoint {k}: no samples recorded"); continue
                np.save(os.path.join(settings.PROJECT_DIR, pfx(f"flow_samples_it{k}.npy")), arr)
            print(f"Saved checkpoint samples at iterations {sorted(set(int(k) for k in args.checkpoints))}")

    # Plot parameter distributions
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        import pandas as pd
        
        df = pd.DataFrame(samples_np, columns=full_param_names)
        # library terms only: terms disabled for this run (and E unless fitted) are left out; pruned ones are shown
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "plots"))
        from plot_param_distributions import plot_param_distributions
        disabled_names = {full_param_names_master[i] for i in global_disabled if i < len(full_param_names_master)}
        shown = [n for n in full_param_names if n not in disabled_names]
        plot_path = os.path.join(out_dir, pfx(f"distributions_{args.material_model}.pdf"))
        plot_param_distributions(samples_np, list(full_param_names), shown, list(model.parameter_names), true_params,
                                 plot_path)
        print(f"Saved parameter distributions plot to {plot_path}")
    except Exception as e:
        print(f"Error plotting distribution histograms: {e}")

    # Generate correlation pairplot for active parameters using matplotlib
    try:
        
        # Filter df to only include active parameters
        active_params_list = [col for col in full_param_names if col in model.parameter_names]
        
        def generate_pairplot(param_names, suffix):
            if len(param_names) > 1:
                print(f"Generating parameter correlation pairplot for {suffix} parameters...")
                df_subset = df[param_names]
                df_sample = df_subset.sample(n=min(1000, len(df_subset)), random_state=42)
                
                n_params = len(param_names)
                fig, axes = plt.subplots(n_params, n_params, figsize=(n_params*2.5, n_params*2.5))
                
                for i in range(n_params):
                    for j in range(n_params):
                        ax = axes[i, j]
                        col_i = param_names[i]
                        col_j = param_names[j]
                        
                        if i < j:
                            ax.set_visible(False)
                        elif i == j:
                            data = df_sample[col_i].values
                            ax.hist(data, bins=30, color='#16a085', alpha=0.7, density=True, edgecolor='white', linewidth=0.5)
                            
                            mean_val = data.mean()
                            true_val = true_params.get(col_i, 0.0)
                            ci_lower = np.percentile(data, 2.5)
                            ci_upper = np.percentile(data, 97.5)
                            
                            ax.axvline(mean_val, color='red', linestyle='-', lw=1.5, alpha=0.8)
                            ax.axvline(true_val, color='black', linestyle='--', lw=1.5, alpha=0.8)
                            ax.axvline(ci_lower, color='red', linestyle=':', lw=1.5, alpha=0.8)
                            ax.axvline(ci_upper, color='red', linestyle=':', lw=1.5, alpha=0.8)
                            
                            title_str = f"True: {true_val:.3f} | Mean: {mean_val:.3f}\n95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]"
                            ax.set_title(title_str, fontsize=9, fontweight='bold', pad=4)
                            
                            if i == n_params - 1:
                                ax.set_xlabel(col_j, fontsize=10, fontweight='bold')
                            else:
                                ax.set_xticklabels([])
                                
                            ax.set_yticks([])
                            if j == 0 and n_params > 1:
                                ax.set_ylabel(col_i, fontsize=10, fontweight='bold')
                        else:
                            ax.scatter(df_sample[col_j], df_sample[col_i], alpha=0.5, s=15, color='#2980b9', edgecolors='none')
                            
                            std_i = df_sample[col_i].std()
                            std_j = df_sample[col_j].std()
                            if std_i == 0 or std_j == 0:
                                corr = 0.0
                            else:
                                corr = df_sample[col_j].corr(df_sample[col_i])
                                if np.isnan(corr): corr = 0.0
                            ax.annotate(f"r = {corr:.3f}", xy=(0.95, 0.95), xycoords='axes fraction', 
                                        ha='right', va='top', fontsize=10, fontweight='bold',
                                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.7))
                                        
                            if i == n_params - 1:
                                ax.set_xlabel(col_j, fontsize=10, fontweight='bold')
                            else:
                                ax.set_xticklabels([])
                            if j == 0:
                                ax.set_ylabel(col_i, fontsize=10, fontweight='bold')
                            else:
                                ax.set_yticklabels([])
                                
                fig.suptitle(f"Parameter Correlation Pairplot ({args.material_model})", y=1.02, fontsize=16, fontweight='bold')
                plt.tight_layout()
                
                pairplot_path = os.path.join(out_dir, pfx(f"parameter_correlation_{args.material_model}_{suffix}.pdf"))
                fig.savefig(pairplot_path, dpi=200, bbox_inches='tight')
                plt.close()
                print(f"Saved parameter correlation pairplot to {pairplot_path}")
            else:
                print(f"Not enough parameters to generate a correlation pairplot for {suffix}.")

        generate_pairplot(list(full_param_names), "all")
        generate_pairplot(active_params_list, "active")
            
    except Exception as e:
        print(f"Error generating parameter correlation pairplot: {e}")

    # Run validation plot scripts (only if not sef_split, otherwise we wait for both to finish in bash script)
    if args.distill_target != "sef_split":
        try:
            subprocess.run(["python3", "plots/plot_distilled_validation.py", 
                            "--distilled_dir", out_dir, 
                            "--material_model", args.material_model,
                            "--distill_target", args.distill_target], check=True)
        except Exception as e:
            print(f"Error running validation plots: {e}")

    # Run invariant sensitivity plot scripts
    try:
        active_params_str = ",".join(model.parameter_names)
        
        args_tail = ["--distilled_dir", out_dir, "--active_params", active_params_str]
        if args.distill_target == "sef_split":
            args_tail.extend(["--component", args.component, "--distill_target", args.distill_target])
            
        subprocess.run(["python3", "plots/plot_invariant_sensitivity.py"] + args_tail, check=True)
        subprocess.run(["python3", "plots/plot_invariant_sensitivity_3d_pairs.py"] + args_tail, check=True)
        subprocess.run(["python3", "plots/plot_deformation_sensitivity.py"] + args_tail, check=True)
        
        # Combined invariant sensitivity (dev vs I1,I2; vol vs J)
        subprocess.run(["python3", "plots/plot_combined_invariant_sensitivity.py", "--distilled_dir", out_dir], check=False)
        # Unified all-invariant sensitivity scatter (dev in blue, vol in orange, aniso in pink across all invariants)
        subprocess.run(["python3", "plots/plot_all_invariant_sensitivity.py", "--distilled_dir", out_dir], check=False)

    except Exception as e:
        print(f"Error running invariant sensitivity plots: {e}")

    pipeline_total_time = time.time() - pipeline_start_time
    print("\n========================================================================")
    print("=== Distillation Pipeline Execution Time Summary ===")
    print("========================================================================")
    for stage_name, duration in time_records.items():
        print(f" * {stage_name:<40}: {duration / 60:>6.2f} min ({duration:>7.2f} sec)")
    print(f" * {'Total Pipeline Duration':<40}: {pipeline_total_time / 60:>6.2f} min ({pipeline_total_time:>7.2f} sec)")
    print("========================================================================\n")

    print("\nDistillation & Sensitivity Analysis Pipeline Finished Successfully!")

if __name__ == "__main__":
    main()
