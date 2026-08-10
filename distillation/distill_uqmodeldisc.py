import sys
import os
import argparse
import numpy as np
import datetime
from pathlib import Path
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'UQInModelDiscovery')))

import torch
import torch.nn as nn

from uqmodeldisc.parameterdistillation.distillation import (
    distill_parameter_distribution_from_gp,
    save_normalizing_flow_parameter_distribution,
    load_normalizing_flow_parameter_distribution,
)
from uqmodeldisc.io import ProjectDirectory
from uqmodeldisc.models import OutputSelectorProtocol
from uqmodeldisc.testcases import (
    TestCases,
    test_case_identifier_uniaxial_tension,
    test_case_identifier_equibiaxial_tension,
    test_case_identifier_pure_shear,
)
from uqmodeldisc.customtypes import Device
from uqmodeldisc.models.base import (
    init_parameter_mask,
    init_parameter_population_matrix,
    update_parameter_population_matrix,
    mask_parameters,
    count_active_parameters,
    filter_active_parameter_indices,
    filter_active_parameter_names,
    filter_active_parameter_scales,
    mask_and_populate_parameters,
    map_parameter_names_to_indices,
)

def plot_loss_monitoring(npz_path, title_suffix, output_path, window_size=100):
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        if not os.path.exists(npz_path):
            print(f"Loss file {npz_path} not found, skipping loss monitoring plot.")
            return
        data = np.load(npz_path)
        w1_loss = data['w1']
        lip_loss = data['lipschitz']
        iterations = np.arange(1, len(w1_loss) + 1)
        
        df = pd.DataFrame({'w1': w1_loss, 'lip': lip_loss})
        w1_ma = df['w1'].rolling(window=window_size, min_periods=1).mean().values
        w1_var = df['w1'].rolling(window=window_size, min_periods=1).var().fillna(0).values
        lip_ma = df['lip'].rolling(window=window_size, min_periods=1).mean().values
        lip_var = df['lip'].rolling(window=window_size, min_periods=1).var().fillna(0).values

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        
        # Plot 1: Wasserstein Distance
        ax1.plot(iterations, w1_loss, color='#bdc3c7', alpha=0.4, label='W1 Distance (Raw)')
        ax1.plot(iterations, w1_ma, color='#2980b9', linewidth=2, label=f'W1 Distance (MA, window={window_size})')
        ax1_right = ax1.twinx()
        ax1_right.plot(iterations, w1_var, color='#e74c3c', linestyle='--', alpha=0.75, label=f'W1 Variance (window={window_size})')
        ax1.set_ylabel('Wasserstein Distance (W1)', fontsize=12, fontweight='bold', color='#2980b9')
        ax1_right.set_ylabel('Variance / Oscillation', fontsize=11, fontweight='bold', color='#e74c3c')
        ax1.set_title(f"Wasserstein Loss & Oscillation Monitoring {title_suffix}", fontsize=14, fontweight='bold')
        ax1.grid(True, linestyle=':', alpha=0.6)
        
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax1_right.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=10)

        # Plot 2: Lipschitz Critic Function Loss
        ax2.plot(iterations, lip_loss, color='#d2b4de', alpha=0.4, label='Lipschitz Function Loss (Raw)')
        ax2.plot(iterations, lip_ma, color='#8e44ad', linewidth=2, label=f'Lipschitz Loss (MA, window={window_size})')
        ax2_right = ax2.twinx()
        ax2_right.plot(iterations, lip_var, color='#e67e22', linestyle='--', alpha=0.75, label=f'Lipschitz Variance (window={window_size})')
        ax2.set_xlabel('Iteration', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Lipschitz Function Loss', fontsize=12, fontweight='bold', color='#8e44ad')
        ax2_right.set_ylabel('Variance / Oscillation', fontsize=11, fontweight='bold', color='#e67e22')
        ax2.set_title(f"Lipschitz Critic Loss & Oscillation Monitoring {title_suffix}", fontsize=14, fontweight='bold')
        ax2.grid(True, linestyle=':', alpha=0.6)

        lines3, labels3 = ax2.get_legend_handles_labels()
        lines4, labels4 = ax2_right.get_legend_handles_labels()
        ax2.legend(lines3 + lines4, labels3 + labels4, loc='upper right', fontsize=10)

        plt.tight_layout()
        plt.savefig(output_path, dpi=200)
        plt.close()
        print(f"Saved loss monitoring plot to {output_path}")
    except Exception as e:
        print(f"Error generating loss monitoring plot: {e}")

class MockGP(nn.Module):
    def __init__(self, loc, cov, device):
        super().__init__()
        self.loc = loc.to(device)
        self.cov = cov.to(device)
        self.likelihood = nn.Module()
    def forward(self, inputs):
        return torch.distributions.MultivariateNormal(self.loc, self.cov)
    def train(self, mode=True):
        pass

class EnergyOutputSelector:
    def __init__(self, num_outputs):
        self.total_num_selected_outputs = num_outputs
    def __call__(self, outputs):
        return outputs

class PyTorchGMRModel(nn.Module):
    def __init__(self, num_points, device, distill_target="sef", include_log_terms=True):
        super().__init__()
        self.distill_target = distill_target
        self.include_log_terms = include_log_terms
        self._output_dim = 4 if distill_target in ["sef_stress", "sef_cauchy"] else 1
        if self.include_log_terms:
            self._num_parameters = 14
            self._parameter_names = ("C10", "C01", "C20", "C11", "C02", "C30", "C21", "C12", "C03", "CL1", "CL2", "D1", "D2", "D3")
        else:
            self._num_parameters = 12
            self._parameter_names = ("C10", "C01", "C20", "C11", "C02", "C30", "C21", "C12", "C03", "D1", "D2", "D3")
        self._parameter_scales = torch.ones(self._num_parameters, device=device)
        self._device = device
        self._num_points = num_points
        self._parameter_mask = init_parameter_mask(self._num_parameters, device)
        self._parameter_population_matrix = init_parameter_population_matrix(self._num_parameters, device)

    @property
    def output_dim(self) -> int: return self._output_dim
    @property
    def num_parameters(self) -> int: return self._num_parameters
    @num_parameters.setter
    def num_parameters(self, val: int): self._num_parameters = val
    @property
    def parameter_names(self): return tuple(self._parameter_names)
    @parameter_names.setter
    def parameter_names(self, val): self._parameter_names = tuple(val)
    @property
    def parameter_scales(self): return self._parameter_scales
    @parameter_scales.setter
    def parameter_scales(self, val): self._parameter_scales = val

    def __call__(self, inputs, test_cases, parameters, validate_args=False):
        # inputs: [batch, 3, 3] F tensor
        # parameters: [num_active_parameters]
        full_parameters = mask_and_populate_parameters(
            parameters,
            self._parameter_mask,
            self._parameter_population_matrix,
        )
        
        # Calculate invariants I1, I2, J
        C = torch.matmul(inputs.transpose(1, 2), inputs)
        I1 = torch.diagonal(C, dim1=1, dim2=2).sum(-1)
        I2 = 0.5 * (I1**2 - torch.diagonal(torch.matmul(C, C), dim1=1, dim2=2).sum(-1))
        J = torch.det(inputs)

        I1_bar = J**(-2/3) * I1
        I2_bar = J**(-4/3) * I2

        I1_m3 = I1_bar - 3.0
        I2_m3 = I2_bar - 3.0
        J_m1 = J - 1.0

        if self.include_log_terms:
            C10, C01, C20, C11, C02, C30, C21, C12, C03, CL1, CL2, D1, D2, D3 = full_parameters
            log1 = torch.log(torch.clamp(I1_bar / 3.0, min=1e-8))
            log2 = torch.log(torch.clamp(I2_bar / 3.0, min=1e-8))
            W_log = CL1 * log1 + CL2 * log2
        else:
            C10, C01, C20, C11, C02, C30, C21, C12, C03, D1, D2, D3 = full_parameters
            W_log = 0.0

        W_dev = (C10 * I1_m3 + C01 * I2_m3 +
                 C20 * I1_m3**2 + C11 * I1_m3 * I2_m3 + C02 * I2_m3**2 +
                 C30 * I1_m3**3 + C21 * (I1_m3**2) * I2_m3 + C12 * I1_m3 * (I2_m3**2) + C03 * I2_m3**3 +
                 W_log)
        
        W_vol = D1 * J_m1**2 + D2 * J_m1**4 + D3 * J_m1**6

        energy = W_dev + W_vol
        if self.distill_target in ["sef_stress", "sef_cauchy"]:
            F_inv_T = torch.linalg.inv(inputs).transpose(1, 2)
            dJ_dF = J.view(-1, 1, 1) * F_inv_T
            dI1bar_dF = J.view(-1, 1, 1)**(-2/3) * (2.0 * inputs - (2.0/3.0) * I1.view(-1, 1, 1) * F_inv_T)
            dI2_dF = 2.0 * I1.view(-1, 1, 1) * inputs - 2.0 * torch.matmul(inputs, torch.matmul(inputs.transpose(1, 2), inputs))
            dI2bar_dF = J.view(-1, 1, 1)**(-4/3) * (dI2_dF - (4.0/3.0) * I2.view(-1, 1, 1) * F_inv_T)
            
            dW_dI1 = C10 + 2.0 * C20 * I1_m3 + C11 * I2_m3 + 3.0 * C30 * (I1_m3**2) + 2.0 * C21 * I1_m3 * I2_m3 + C12 * (I2_m3**2)
            dW_dI2 = C01 + C11 * I1_m3 + 2.0 * C02 * I2_m3 + C21 * (I1_m3**2) + 2.0 * C12 * I1_m3 * I2_m3 + 3.0 * C03 * (I2_m3**2)
            if self.include_log_terms:
                dW_dI1 = dW_dI1 + CL1 / I1_bar
                dW_dI2 = dW_dI2 + CL2 / I2_bar
            dW_dJ = 2.0 * D1 * J_m1 + 4.0 * D2 * (J_m1**3) + 6.0 * D3 * (J_m1**5)
            
            stress = dW_dI1.view(-1, 1, 1) * dI1bar_dF + dW_dI2.view(-1, 1, 1) * dI2bar_dF + dW_dJ.view(-1, 1, 1) * dJ_dF
            if self.distill_target == "sef_cauchy":
                stress = torch.matmul(stress, inputs.transpose(1, 2)) / J.view(-1, 1, 1)
            p00 = stress[:, 0, 0]
            p11 = stress[:, 1, 1]
            p01 = stress[:, 0, 1]
            return torch.stack([energy, p00, p11, p01], dim=-1) # shape [num_points, 4]
        else:
            return energy.unsqueeze(-1) # shape [num_points, 1]
        
    def deactivate_parameters(self, indices):
        mask_parameters(indices, self._parameter_mask, False)

    def deactivate_all_parameters(self):
        self.deactivate_parameters(list(range(self.num_parameters)))

    def activate_parameters(self, indices):
        mask_parameters(indices, self._parameter_mask, True)

    def reset_parameter_deactivations(self):
        self._parameter_mask = init_parameter_mask(self.num_parameters, self._device)

    def get_active_parameter_indices(self):
        return filter_active_parameter_indices(self._parameter_mask)

    def get_active_parameter_names(self):
        return filter_active_parameter_names(self._parameter_mask, self.parameter_names)

    def get_number_of_active_parameters(self):
        return count_active_parameters(self._parameter_mask)

    def reduce_to_activated_parameters(self):
        old_mask = self._parameter_mask
        self.num_parameters = self.get_number_of_active_parameters()
        self.parameter_names = self.get_active_parameter_names()
        self.parameter_scales = filter_active_parameter_scales(self._parameter_mask, self.parameter_scales)
        self._parameter_mask = init_parameter_mask(self.num_parameters, self._device)
        self._parameter_population_matrix = update_parameter_population_matrix(
            self._parameter_population_matrix, old_mask
        )

    def reduce_model_to_parameter_names(self, names):
        indices = map_parameter_names_to_indices(names, self.parameter_names)
        self.deactivate_all_parameters()
        self.activate_parameters(indices)
        self.reduce_to_activated_parameters()

    def get_model_state(self):
        return self._parameter_population_matrix, self.parameter_scales

    def init_model_state(self, pop, scales):
        self._parameter_population_matrix = pop
        self.parameter_scales = scales


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--saved_model_dir", type=str, required=True)
    parser.add_argument("--material_model", type=str, default="gmr", help="Material model to distill (gmr/gmr_log with log terms, gmr_nolog without log terms)")
    parser.add_argument("--n_iterations", type=int, default=5000)
    parser.add_argument("--load_distilled_dir", type=str, default=None, help="Path to existing distilled model directory to load flow parameters without re-running Stage 1")
    parser.add_argument("--do_sensitivity", action="store_true", default=True, help="Perform Sobol sensitivity analysis and re-distill with sensitive parameters")
    parser.add_argument("--no_sensitivity", dest="do_sensitivity", action="store_false", help="Skip sensitivity analysis")
    parser.add_argument("--sobol_threshold", type=float, default=1e-4, help="Total Sobol index threshold for selecting sensitive parameters")
    parser.add_argument("--sobol_samples_factor", type=int, default=1024, help="Saltelli sample factor for sensitivity analysis")
    parser.add_argument("--sample_mode", type=str, default="dataset_f", choices=["standard", "standard_interp", "dataset_f", "dataset_all"], help="Sample deformation inputs from extraction dataset directly or standard modes (with or without interpolation clipping)")
    parser.add_argument("--num_points", type=int, default=192, help="Number of points for GP joint evaluation and distillation")
    parser.add_argument("--max_gamma", type=float, default=1.0, help="Max deformation intensity gamma when sample_mode is standard")
    parser.add_argument("--distill_target", type=str, default="sef", choices=["sef", "sef_stress", "sef_cauchy"], help="Target mode: strain energy function (sef), joint SEF + Piola stress (sef_stress), or joint SEF + Cauchy stress (sef_cauchy)")
    parser.add_argument("--load_existing_sensitivities", action="store_true", help="Skip Sobol resampling and directly load existing sensitivity CSVs from out_dir")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    torch.set_default_dtype(torch.float64)

    if args.sample_mode == "dataset_all":
        export_subfolder = "pytorch_export_dataset_all"
    elif args.sample_mode == "dataset_f":
        export_subfolder = "pytorch_export_dataset_f"
    elif args.sample_mode == "standard_interp":
        export_subfolder = "pytorch_export_standard_interp"
    else:
        export_subfolder = f"pytorch_export_standard_g{args.max_gamma}"
    if args.distill_target in ["sef_stress", "sef_cauchy"]:
        export_subfolder += f"_{args.distill_target}"
        
    export_dir = os.path.join(args.saved_model_dir, export_subfolder)
    if not os.path.exists(export_dir) or not os.path.exists(os.path.join(export_dir, "mean_psi.npy")):
        import subprocess
        print(f"'{export_dir}' not found. Exporting GP to PyTorch first (sample_mode: {args.sample_mode}, max_gamma: {args.max_gamma}, distill_target: {args.distill_target})...")
        subprocess.run(["python3", "distillation/export_gp_to_pytorch.py", "--saved_model_dir", args.saved_model_dir, "--sample_mode", args.sample_mode, "--num_points", str(args.num_points), "--max_gamma", str(args.max_gamma), "--distill_target", args.distill_target, "--export_subfolder", export_subfolder], check=True)
        
    mean_psi = torch.tensor(np.load(os.path.join(export_dir, "mean_psi.npy")), dtype=torch.float64, device=device)
    cov_psi = torch.tensor(np.load(os.path.join(export_dir, "cov_psi.npy")), dtype=torch.float64, device=device)
    cov_psi = (cov_psi + cov_psi.T) / 2.0
    # Ensure strict positive-definiteness without artificial diagonal inflation (eigenvalue clipping)
    L, V = torch.linalg.eigh(cov_psi)
    if (L <= 1e-8).any():
        min_eig = L.min().item()
        print(f"Adjusting non-positive numerical eigenvalues (min eigen: {min_eig:.3e}) to ensure positive-definiteness without diagonal jitter...")
        L = torch.clamp(L, min=1e-8)
        cov_psi = V @ torch.diag(L) @ V.T
        cov_psi = (cov_psi + cov_psi.T) / 2.0
        try:
            torch.linalg.cholesky(cov_psi)
        except RuntimeError:
            cov_psi += 1e-6 * torch.eye(cov_psi.shape[0], dtype=torch.float64, device=device)
    f3x3 = torch.tensor(np.load(os.path.join(export_dir, "f3x3.npy")), dtype=torch.float64, device=device)

    num_points = f3x3.shape[0]
    
    # Map deformation modes to test case identifiers for sensitivity analysis output grouping
    test_cases = torch.zeros(num_points, dtype=torch.int64, device=device)
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

    mock_gp = MockGP(mean_psi, cov_psi, device=device)
    include_log = args.material_model not in ["gmr_nolog", "gmr_no_log"]
    model = PyTorchGMRModel(num_points, device=device, distill_target=args.distill_target, include_log_terms=include_log)
    full_param_names_master = model.parameter_names
    output_selector = EnergyOutputSelector(num_outputs=mean_psi.shape[0])

    from uqmodeldisc.settings import Settings
    settings = Settings()
    
    if args.load_distilled_dir and os.path.exists(args.load_distilled_dir):
        out_dir = os.path.abspath(args.load_distilled_dir)
        log_mode = "a"
    else:
        model_folder_name = os.path.basename(os.path.normpath(args.saved_model_dir))
        parts = model_folder_name.split('_')
        
        true_model_name = "isihara" # We can just hardcode or infer it
        for p in ["isihara", "nh", "neohookean2", "nh2", "gentthomas", "nh4", "neohookean4"]:
            if p in parts:
                true_model_name = p
                break
                
        true_model = get_material(true_model_name, jit_P=False)
        
        true_params = {}
        if true_model_name == "isihara":
            true_params = {"C10": true_model.c10, "C01": true_model.c01, "C20": true_model.c20, "D1": true_model.d1}
        elif true_model_name in ["nh", "neohookean2", "nh2"]:
            true_params = {"C10": true_model.dev_params[0], "D1": true_model.vol_params[0]}
        elif true_model_name in ["nh4", "neohookean4"]:
            true_params = {"C10": true_model.dev_params[0], "D2": true_model.vol_params[1]}
        elif true_model_name in ["gentthomas"]:
            true_params = {"C10": true_model.dev_params[0], "CL2": true_model.dev_params[10], "D1": true_model.vol_params[0]}
        source_type = "exp" if any(x in model_folder_name.lower() for x in ["exp", "dic", "18617429"]) else "syn"
        noise_str = f"_{source_type}"
        if len(parts) >= 4:
            try:
                _ = float(parts[2])
                _ = float(parts[3])
                noise_str = f"_{source_type}_d{parts[2]}_l{parts[3]}"
            except ValueError:
                pass
        
        current_time = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        mode_str = f"_{args.sample_mode}_g{args.max_gamma}" if args.sample_mode == "standard" else f"_{args.sample_mode}"
        if args.distill_target in ["sef_stress", "sef_cauchy"]:
            mode_str += f"_{args.distill_target}"
        out_dir = os.path.abspath(os.path.join("distillation", "distilled_models", f"{current_time}_{ugp_model_name}{noise_str}_{args.material_model}{mode_str}_uqmodeldisc"))
        log_mode = "w"
    os.makedirs(out_dir, exist_ok=True)
    
    with open(os.path.join(out_dir, "source_extraction_dir.txt"), "w") as f:
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
            
    sys.stdout = TeeLogger(os.path.join(out_dir, "distillation_log.txt"), mode=log_mode)
    print(f"\nLogging distillation outputs to {os.path.join(out_dir, 'distillation_log.txt')}")
    
    settings.PROJECT_DIR = Path(out_dir)
    project_dir = ProjectDirectory(settings)
    time_records = {}
    pipeline_start_time = time.time()

    if args.load_distilled_dir and os.path.exists(os.path.join(out_dir, "output", "normalizing_flow_parameters")):
        print("\n========================================================================")
        print("=== Stage 1: Loading Existing Distilled Model (Skipping Training) ===")
        print("========================================================================\n")
        distribution = load_normalizing_flow_parameter_distribution(
            model=model,
            output_subdirectory=".",
            project_directory=project_dir,
            device=device
        )
    else:
        print("\n========================================================================")
        print("=== Stage 1: Initial Parameter Distillation (Full Model) ===")
        print("========================================================================\n")
        t0_stage1 = time.time()

        distribution = distill_parameter_distribution_from_gp(
            gp=mock_gp,
            model=model,
            output_selector=output_selector,
            distribution_type="normalizing flow",
            inputs=f3x3,
            test_cases=test_cases,
            num_func_samples=512,
            lipschitz_penalty_coefficient=10.0,
            num_iters_wasserstein=args.n_iterations,
            num_layers_lipschitz_nn=3,
            relative_width_lipschitz_nn=1.0,
            num_iters_lipschitz=10,
            output_subdirectory=".",
            project_directory=project_dir,
            device=device,
            stage_tag="before_sensitivity",
        )
        t_stage1_dur = time.time() - t0_stage1
        time_records['Stage 1 (Initial Distillation)'] = t_stage1_dur
        print(f"\n[Time Usage] Stage 1 finished in {t_stage1_dur / 60:.2f} minutes ({t_stage1_dur:.2f} seconds).\n")
        
        save_normalizing_flow_parameter_distribution(
            distribution=distribution,
            output_subdirectory=".",
            project_directory=project_dir,
            device=device
        )

        plot_loss_monitoring(
            os.path.join(out_dir, "loss_history_before_sensitivity.npz"),
            title_suffix="(Stage 1: Before Sensitivity Analysis)",
            output_path=os.path.join(out_dir, "loss_monitoring_before_sensitivity.png"),
            window_size=min(100, max(10, args.n_iterations // 50))
        )

    # Save parameter distribution plot immediately before running sensitivity analysis
    try:
        print("\nSaving parameter distribution plot before sensitivity analysis...")
        with torch.no_grad():
            samples_pre = distribution.sample(5000).cpu().numpy()
        full_param_names_pre = full_param_names_master
        pre_samples_path = os.path.join(out_dir, "flow_samples_before_sensitivity.npy")
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
            axes_pre[i].set_title(f"{col} (Pre-Sensitivity)\nMean: {mean_val:.4f}", color='#2c3e50', fontsize=12, fontweight='bold')
            axes_pre[i].legend(fontsize=9)
        for j in range(len(full_param_names_pre), len(axes_pre)):
            axes_pre[j].set_visible(False)
        plt.tight_layout()
        pre_plot_path = os.path.join(out_dir, f"distributions_before_sensitivity_{args.material_model}.png")
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
            csv_dir = os.path.join(out_dir, "output", "sensitivities")
            if not os.path.exists(os.path.join(csv_dir, "total_sobol_indices_statistics_output_0.csv")):
                csv_dir = sensitivities_dir
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
                data_set_label="treloar",
                inputs=f3x3,
                test_cases=test_cases,
                select_model=True,
                total_sobol_index_thresshold=args.sobol_threshold,
                output_subdirectory="sensitivities",
                project_directory=project_dir,
                device=device,
            )

        # Generate custom bar plot for Sobol indices
        try:
            import matplotlib.pyplot as plt
            import pandas as pd
            csv_dir = os.path.join(out_dir, "output", "sensitivities")
            if not os.path.exists(os.path.join(csv_dir, "total_sobol_indices_statistics_output_0.csv")):
                csv_dir = os.path.join(out_dir, "sensitivities")
            stats_path = os.path.join(csv_dir, "total_sobol_indices_statistics_output_0.csv")
            first_stats_path = os.path.join(csv_dir, "first_sobol_indices_statistics_output_0.csv")
            if os.path.exists(stats_path) and os.path.exists(first_stats_path):
                df_tot = pd.read_csv(stats_path).fillna(0.0)
                df_first = pd.read_csv(first_stats_path).fillna(0.0)
                
                param_cols = [c for c in df_tot.columns if c not in ["Unnamed: 0", ""]]
                tot_means = df_tot.iloc[0][param_cols].values.astype(float)
                tot_stds = df_tot.iloc[1][param_cols].values.astype(float)
                first_means = df_first.iloc[0][param_cols].values.astype(float)
                first_stds = df_first.iloc[1][param_cols].values.astype(float)
                
                x = np.arange(len(param_cols))
                width = 0.35
                
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.bar(x - width/2, first_means, width, yerr=first_stds, label="First-Order ($S_1$)", capsize=4, color="#3498db", alpha=0.85)
                ax.bar(x + width/2, tot_means, width, yerr=tot_stds, label="Total-Order ($S_T$)", capsize=4, color="#e74c3c", alpha=0.85)
                
                ax.set_yscale('log')
                ax.set_ylim(bottom=1e-5)
                
                ax.axhline(args.sobol_threshold, color='black', linestyle='--', linewidth=1.5, label=f"Threshold ({args.sobol_threshold})")
                ax.set_ylabel('Sobol Sensitivity Index', fontsize=12, fontweight='bold')
                ax.set_title('Material Parameter Sensitivity (Sobol Indices)', fontsize=14, fontweight='bold')
                ax.set_xticks(x)
                ax.set_xticklabels(param_cols, fontsize=11, fontweight='bold')
                ax.legend(fontsize=11)
                ax.grid(axis='y', linestyle=':', alpha=0.6)
                plt.tight_layout()
                sobol_plot_path = os.path.join(out_dir, "sobol_sensitivity_indices.png")
                plt.savefig(sobol_plot_path, dpi=200)
                plt.close()
                print(f"Saved Sobol sensitivity bar plot to {sobol_plot_path}")
        except Exception as e:
            print(f"Error plotting Sobol indices: {e}")

        t_stage2_dur = time.time() - t0_stage2
        time_records['Stage 2 (Sobol Sensitivity Analysis)'] = t_stage2_dur
        print(f"\n[Time Usage] Stage 2 finished in {t_stage2_dur / 60:.2f} minutes ({t_stage2_dur:.2f} seconds).\n")

        active_names = model.get_active_parameter_names()
        print(f"\nSelected sensitive parameters ({len(active_names)}): {active_names}")
        if len(active_names) == 0:
            print("Warning: No parameters exceeded Sobol threshold! Keeping all parameters active.")
            model.activate_parameters(list(range(model.num_parameters)))
        elif len(active_names) < model.num_parameters:
            print("\n========================================================================")
            print("=== Stage 3: Re-distillation with Sensitive Parameters Only ===")
            print("========================================================================\n")
            model.reduce_to_activated_parameters()
            print(f"Reduced model active parameters ({model.num_parameters}): {model.parameter_names}")
            t0_stage3 = time.time()
            
            distribution = distill_parameter_distribution_from_gp(
                gp=mock_gp,
                model=model,
                output_selector=output_selector,
                distribution_type="normalizing flow",
                inputs=f3x3,
                test_cases=test_cases,
                num_func_samples=512,
                lipschitz_penalty_coefficient=10.0,
                num_iters_wasserstein=args.n_iterations,
                num_layers_lipschitz_nn=3,
                relative_width_lipschitz_nn=1.0,
                num_iters_lipschitz=10,
                output_subdirectory=".",
                project_directory=project_dir,
                device=device,
                stage_tag="after_sensitivity",
            )
            t_stage3_dur = time.time() - t0_stage3
            time_records['Stage 3 (Re-Distillation Pruned Model)'] = t_stage3_dur
            print(f"\n[Time Usage] Stage 3 finished in {t_stage3_dur / 60:.2f} minutes ({t_stage3_dur:.2f} seconds).\n")
            
            plot_loss_monitoring(
                os.path.join(out_dir, "loss_history_after_sensitivity.npz"),
                title_suffix="(Stage 3: After Sensitivity Analysis)",
                output_path=os.path.join(out_dir, "loss_monitoring_after_sensitivity.png"),
                window_size=min(100, max(10, args.n_iterations // 50))
            )
        else:
            print("All material parameters were determined to be sensitive! No model reduction needed.")

    # Save the PyTorch Normalizing Flow model weights
    save_normalizing_flow_parameter_distribution(
        distribution=distribution,
        output_subdirectory=".",
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
            
        samples_path = os.path.join(settings.PROJECT_DIR, "flow_samples.npy")
        np.save(samples_path, samples_np)
        print(f"Saved distilled PyTorch parameter samples (shape: {samples_np.shape}) to {samples_path}")

    # Plot parameter distributions
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        import pandas as pd
        
        df = pd.DataFrame(samples_np, columns=full_param_names)
        num_params_len = len(full_param_names)
        cols = 4
        rows = (num_params_len + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*4))
        axes = axes.flatten()
        
        means = df.mean()
        for i, col in enumerate(full_param_names):
            is_active = col in model.parameter_names
            if is_active:
                sns.histplot(df[col], ax=axes[i], color='#2980b9', bins=30, kde=True, edgecolor="black", alpha=0.7)
                mean_val = means[col]
                axes[i].axvline(mean_val, color='#c0392b', linestyle='--', linewidth=2, label=f"Mean: {mean_val:.4f}")
                axes[i].set_title(f"{col}\nMean: {mean_val:.4f}", color='#2c3e50', fontsize=12, fontweight='bold')
                axes[i].legend(fontsize=9)
            else:
                sns.histplot([0], ax=axes[i], color='gray', bins=1)
                axes[i].set_title(f"{col}\n[Deactivated by Sensitivity]", color='#7f8c8d', fontsize=12, style='italic')
                axes[i].set_xlabel("Value (Fixed to 0)")
            
        for j in range(num_params_len, len(axes)):
            axes[j].set_visible(False)
            
        plt.tight_layout()
        plot_path = os.path.join(out_dir, f"distributions_{args.material_model}.png")
        plt.savefig(plot_path, dpi=200)
        plt.close()
        print(f"Saved parameter distributions plot to {plot_path}")
    except Exception as e:
        print(f"Error plotting distribution histograms: {e}")

    # Generate correlation pairplot for active parameters using matplotlib
    try:
        from scipy.stats import gaussian_kde
        
        # Filter df to only include active parameters
        active_params_list = [col for col in full_param_names if col in model.parameter_names]
        
        if len(active_params_list) > 1:
            print("Generating parameter correlation pairplot...")
            df_active = df[active_params_list]
            
            # Use a smaller subset of samples if there are too many, to speed up plotting and reduce file size
            df_sample = df_active.sample(n=min(1000, len(df_active)), random_state=42)
            
            n_params = len(active_params_list)
            fig, axes = plt.subplots(n_params, n_params, figsize=(n_params*2.5, n_params*2.5))
            
            for i in range(n_params):
                for j in range(n_params):
                    ax = axes[i, j]
                    col_i = active_params_list[i]
                    col_j = active_params_list[j]
                    
                    if i < j:
                        # Upper triangle - hide
                        ax.set_visible(False)
                    elif i == j:
                        # Diagonal - Histogram
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
                            
                        # Hide y-axis for diagonal (like seaborn does)
                        ax.set_yticks([])
                        if j == 0 and n_params > 1:
                            ax.set_ylabel(col_i, fontsize=10, fontweight='bold')
                    else:
                        # Lower triangle - Scatter
                        ax.scatter(df_sample[col_j], df_sample[col_i], alpha=0.5, s=15, color='#2980b9', edgecolors='none')
                        
                        corr = df_sample[col_j].corr(df_sample[col_i])
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
            
            pairplot_path = os.path.join(out_dir, f"parameter_correlation_{args.material_model}.png")
            fig.savefig(pairplot_path, dpi=200, bbox_inches='tight')
            plt.close()
            print(f"Saved parameter correlation pairplot to {pairplot_path}")
        else:
            print("Not enough active parameters to generate a correlation pairplot.")
            
    except Exception as e:
        print(f"Error generating parameter correlation pairplot: {e}")

    
    # Run validation plot script
    try:
        import subprocess
        subprocess.run(["python3", "plots/plot_distilled_validation.py", 
                        "--distilled_dir", out_dir, 
                        "--material_model", args.material_model], check=True)
    except Exception as e:
        print(f"Error running validation plots: {e}")

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
