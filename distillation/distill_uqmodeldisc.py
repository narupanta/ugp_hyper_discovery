import sys
import os
import argparse
import numpy as np
import datetime
from pathlib import Path
import time
from core.material_models import get_material

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

def plot_loss_monitoring(npz_path, title_suffix, output_path, window_size=100, last_n_iterations=None):
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
        
        has_grad_penalty = 'grad_penalty' in data

        df = pd.DataFrame({'w1': w1_loss, 'lip': lip_loss})
        w1_ma = df['w1'].rolling(window=window_size, min_periods=1).mean().values
        w1_var = df['w1'].rolling(window=window_size, min_periods=1).var().fillna(0).values
        lip_ma = df['lip'].rolling(window=window_size, min_periods=1).mean().values
        lip_var = df['lip'].rolling(window=window_size, min_periods=1).var().fillna(0).values

        if has_grad_penalty:
            grad_penalty = data['grad_penalty']
            df['gp'] = grad_penalty
            gp_ma = df['gp'].rolling(window=window_size, min_periods=1).mean().values
            gp_var = df['gp'].rolling(window=window_size, min_periods=1).var().fillna(0).values

        if last_n_iterations is not None:
            if len(iterations) <= last_n_iterations:
                return
            w1_loss = w1_loss[-last_n_iterations:]
            lip_loss = lip_loss[-last_n_iterations:]
            iterations = iterations[-last_n_iterations:]
            w1_ma = w1_ma[-last_n_iterations:]
            w1_var = w1_var[-last_n_iterations:]
            lip_ma = lip_ma[-last_n_iterations:]
            lip_var = lip_var[-last_n_iterations:]
            if has_grad_penalty:
                grad_penalty = grad_penalty[-last_n_iterations:]
                gp_ma = gp_ma[-last_n_iterations:]
                gp_var = gp_var[-last_n_iterations:]

        nrows = 3 if has_grad_penalty else 2
        figsize = (12, 15) if has_grad_penalty else (12, 10)
        
        fig, axes = plt.subplots(nrows, 1, figsize=figsize, sharex=True)
        if nrows == 2:
            ax1, ax2 = axes
        else:
            ax1, ax2, ax3 = axes
        
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
        if not has_grad_penalty:
            ax2.set_xlabel('Iteration', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Lipschitz Function Loss', fontsize=12, fontweight='bold', color='#8e44ad')
        ax2_right.set_ylabel('Variance / Oscillation', fontsize=11, fontweight='bold', color='#e67e22')
        ax2.set_title(f"Lipschitz Critic Loss & Oscillation Monitoring {title_suffix}", fontsize=14, fontweight='bold')
        ax2.grid(True, linestyle=':', alpha=0.6)

        lines3, labels3 = ax2.get_legend_handles_labels()
        lines4, labels4 = ax2_right.get_legend_handles_labels()
        ax2.legend(lines3 + lines4, labels3 + labels4, loc='upper right', fontsize=10)

        if has_grad_penalty:
            # Plot 3: Gradient Penalty
            ax3.plot(iterations, grad_penalty, color='#f5cba7', alpha=0.4, label='Gradient Penalty (Raw)')
            ax3.plot(iterations, gp_ma, color='#d35400', linewidth=2, label=f'Gradient Penalty (MA, window={window_size})')
            ax3_right = ax3.twinx()
            ax3_right.plot(iterations, gp_var, color='#c0392b', linestyle='--', alpha=0.75, label=f'Penalty Variance (window={window_size})')
            ax3.set_xlabel('Iteration', fontsize=12, fontweight='bold')
            ax3.set_ylabel('Gradient Penalty', fontsize=12, fontweight='bold', color='#d35400')
            ax3_right.set_ylabel('Variance / Oscillation', fontsize=11, fontweight='bold', color='#c0392b')
            ax3.set_title(f"Gradient Penalty Monitoring {title_suffix}", fontsize=14, fontweight='bold')
            ax3.grid(True, linestyle=':', alpha=0.6)
            
            lines5, labels5 = ax3.get_legend_handles_labels()
            lines6, labels6 = ax3_right.get_legend_handles_labels()
            ax3.legend(lines5 + lines6, labels5 + labels6, loc='upper right', fontsize=10)

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

class BasePyTorchMaterialModel(nn.Module):
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


class PyTorchGMRModel(BasePyTorchMaterialModel):
    def __init__(self, num_points, device, include_log_terms=False, distill_target="sef"):
        super().__init__()
        self.include_log_terms = include_log_terms
        self.distill_target = distill_target
        self._output_dim = 4 if distill_target in ["sef_stress", "sef_cauchy"] else 1
        
        if self.include_log_terms:
            self._num_parameters = 13
            self._parameter_names = ("$C_{10}$", "$C_{01}$", "$C_{20}$", "$C_{11}$", "$C_{02}$", "$C_{30}$", "$C_{21}$", "$C_{12}$", "$C_{03}$", "$E$", "$D_{1}$", "$D_{2}$", "$D_{3}$")
        else:
            self._num_parameters = 12
            self._parameter_names = ("$C_{10}$", "$C_{01}$", "$C_{20}$", "$C_{11}$", "$C_{02}$", "$C_{30}$", "$C_{21}$", "$C_{12}$", "$C_{03}$", "$D_{1}$", "$D_{2}$", "$D_{3}$")
        
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
        full_parameters = mask_and_populate_parameters(parameters, self._parameter_mask, self._parameter_population_matrix)
        C = torch.matmul(inputs.transpose(1, 2), inputs)
        I1 = torch.diagonal(C, dim1=1, dim2=2).sum(-1)
        I2 = 0.5 * (I1**2 - torch.diagonal(torch.matmul(C, C), dim1=1, dim2=2).sum(-1))
        J = torch.det(inputs)
        I1_bar, I2_bar = J**(-2/3) * I1, J**(-4/3) * I2
        I1_m3, I2_m3, J_m1 = I1_bar - 3.0, I2_bar - 3.0, J - 1.0

        if self.include_log_terms:
            C10, C01, C20, C11, C02, C30, C21, C12, C03, E, D1, D2, D3 = full_parameters
            log2 = torch.log(torch.clamp(I2_bar / 3.0, min=1e-8))
            W_log = E * log2
        else:
            C10, C01, C20, C11, C02, C30, C21, C12, C03, D1, D2, D3 = full_parameters
            W_log = 0.0

        W_dev = (C10 * I1_m3 + C01 * I2_m3 + C20 * I1_m3**2 + C11 * I1_m3 * I2_m3 + C02 * I2_m3**2 +
                 C30 * I1_m3**3 + C21 * (I1_m3**2) * I2_m3 + C12 * I1_m3 * (I2_m3**2) + C03 * I2_m3**3 + W_log)
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
                dW_dI2 += E / torch.clamp(I2_bar, min=1e-8)
                
            dW_dJ = 2.0 * D1 * J_m1 + 4.0 * D2 * J_m1**3 + 6.0 * D3 * J_m1**5
            P = dW_dI1.view(-1, 1, 1) * dI1bar_dF + dW_dI2.view(-1, 1, 1) * dI2bar_dF + dW_dJ.view(-1, 1, 1) * dJ_dF
            
            if self.distill_target == "sef_cauchy":
                cauchy = (1.0 / J.view(-1, 1, 1)) * torch.matmul(P, inputs.transpose(1, 2))
                return torch.cat([energy.unsqueeze(1), cauchy[:, 0, 0].unsqueeze(1), cauchy[:, 1, 1].unsqueeze(1), cauchy[:, 0, 1].unsqueeze(1)], dim=1)
            else:
                return torch.cat([energy.unsqueeze(1), P[:, 0, 0].unsqueeze(1), P[:, 1, 1].unsqueeze(1), P[:, 0, 1].unsqueeze(1)], dim=1)
        return energy.unsqueeze(-1)


class PyTorchGMRDevModel(BasePyTorchMaterialModel):
    def __init__(self, num_points, device, include_log_terms=False, distill_target="sef"):
        super().__init__()
        self.include_log_terms = include_log_terms
        self.distill_target = distill_target
        self._output_dim = 4 if distill_target in ["sef_stress", "sef_cauchy"] else 1
        
        if self.include_log_terms:
            self._num_parameters = 10
            self._parameter_names = ("$C_{10}$", "$C_{01}$", "$C_{20}$", "$C_{11}$", "$C_{02}$", "$C_{30}$", "$C_{21}$", "$C_{12}$", "$C_{03}$", "$E$")
        else:
            self._num_parameters = 9
            self._parameter_names = ("$C_{10}$", "$C_{01}$", "$C_{20}$", "$C_{11}$", "$C_{02}$", "$C_{30}$", "$C_{21}$", "$C_{12}$", "$C_{03}$")
        
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
        full_parameters = mask_and_populate_parameters(parameters, self._parameter_mask, self._parameter_population_matrix)
        C = torch.matmul(inputs.transpose(1, 2), inputs)
        I1 = torch.diagonal(C, dim1=1, dim2=2).sum(-1)
        I2 = 0.5 * (I1**2 - torch.diagonal(torch.matmul(C, C), dim1=1, dim2=2).sum(-1))
        J = torch.det(inputs)
        I1_bar, I2_bar = J**(-2/3) * I1, J**(-4/3) * I2
        I1_m3, I2_m3 = I1_bar - 3.0, I2_bar - 3.0

        if self.include_log_terms:
            C10, C01, C20, C11, C02, C30, C21, C12, C03, E = full_parameters
            log2 = torch.log(torch.clamp(I2_bar / 3.0, min=1e-8))
            W_log = E * log2
        else:
            C10, C01, C20, C11, C02, C30, C21, C12, C03 = full_parameters
            W_log = 0.0

        W_dev = (C10 * I1_m3 + C01 * I2_m3 + C20 * I1_m3**2 + C11 * I1_m3 * I2_m3 + C02 * I2_m3**2 +
                 C30 * I1_m3**3 + C21 * (I1_m3**2) * I2_m3 + C12 * I1_m3 * (I2_m3**2) + C03 * I2_m3**3 + W_log)
        energy = W_dev

        if self.distill_target in ["sef_stress", "sef_cauchy"]:
            F_inv_T = torch.linalg.inv(inputs).transpose(1, 2)
            dI1bar_dF = J.view(-1, 1, 1)**(-2/3) * (2.0 * inputs - (2.0/3.0) * I1.view(-1, 1, 1) * F_inv_T)
            dI2_dF = 2.0 * I1.view(-1, 1, 1) * inputs - 2.0 * torch.matmul(inputs, torch.matmul(inputs.transpose(1, 2), inputs))
            dI2bar_dF = J.view(-1, 1, 1)**(-4/3) * (dI2_dF - (4.0/3.0) * I2.view(-1, 1, 1) * F_inv_T)
            
            dW_dI1 = C10 + 2.0 * C20 * I1_m3 + C11 * I2_m3 + 3.0 * C30 * (I1_m3**2) + 2.0 * C21 * I1_m3 * I2_m3 + C12 * (I2_m3**2)
            dW_dI2 = C01 + C11 * I1_m3 + 2.0 * C02 * I2_m3 + C21 * (I1_m3**2) + 2.0 * C12 * I1_m3 * I2_m3 + 3.0 * C03 * (I2_m3**2)
            if self.include_log_terms:
                dW_dI2 += E / torch.clamp(I2_bar, min=1e-8)
                
            P = dW_dI1.view(-1, 1, 1) * dI1bar_dF + dW_dI2.view(-1, 1, 1) * dI2bar_dF
            
            if self.distill_target == "sef_cauchy":
                cauchy = (1.0 / J.view(-1, 1, 1)) * torch.matmul(P, inputs.transpose(1, 2))
                return torch.cat([energy.unsqueeze(1), cauchy[:, 0, 0].unsqueeze(1), cauchy[:, 1, 1].unsqueeze(1), cauchy[:, 0, 1].unsqueeze(1)], dim=1)
            else:
                return torch.cat([energy.unsqueeze(1), P[:, 0, 0].unsqueeze(1), P[:, 1, 1].unsqueeze(1), P[:, 0, 1].unsqueeze(1)], dim=1)
        return energy.unsqueeze(-1)


class PyTorchGMRVolModel(BasePyTorchMaterialModel):
    def __init__(self, num_points, device, distill_target="sef"):
        super().__init__()
        self.distill_target = distill_target
        self._output_dim = 4 if distill_target in ["sef_stress", "sef_cauchy"] else 1
        
        self._num_parameters = 3
        self._parameter_names = ("$D_{1}$", "$D_{2}$", "$D_{3}$")
        
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
        full_parameters = mask_and_populate_parameters(parameters, self._parameter_mask, self._parameter_population_matrix)
        J = torch.det(inputs)
        J_m1 = J - 1.0

        D1, D2, D3 = full_parameters

        W_vol = D1 * J_m1**2 + D2 * J_m1**4 + D3 * J_m1**6
        energy = W_vol

        if self.distill_target in ["sef_stress", "sef_cauchy"]:
            F_inv_T = torch.linalg.inv(inputs).transpose(1, 2)
            dJ_dF = J.view(-1, 1, 1) * F_inv_T
                
            dW_dJ = 2.0 * D1 * J_m1 + 4.0 * D2 * J_m1**3 + 6.0 * D3 * J_m1**5
            P = dW_dJ.view(-1, 1, 1) * dJ_dF
            
            if self.distill_target == "sef_cauchy":
                cauchy = (1.0 / J.view(-1, 1, 1)) * torch.matmul(P, inputs.transpose(1, 2))
                return torch.cat([energy.unsqueeze(1), cauchy[:, 0, 0].unsqueeze(1), cauchy[:, 1, 1].unsqueeze(1), cauchy[:, 0, 1].unsqueeze(1)], dim=1)
            else:
                return torch.cat([energy.unsqueeze(1), P[:, 0, 0].unsqueeze(1), P[:, 1, 1].unsqueeze(1), P[:, 0, 1].unsqueeze(1)], dim=1)
        return energy.unsqueeze(-1)


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
    parser.add_argument("--pruning_mode", type=str, default="ec", choices=["ec", "threshold"], help="Pruning mode: 'ec' (Estimated Coverage) or 'threshold' (Fixed Sobol Threshold).")
    parser.add_argument("--ec_threshold", type=float, default=95.0, help="Estimated Coverage threshold in percent (default: 95.0). Only used if pruning_mode='ec'.")
    parser.add_argument("--sample_mode", type=str, default="dataset_f", choices=["standard", "standard_interp", "dataset_f", "dataset_all", "inducing_points"], help="Sample deformation inputs from extraction dataset directly or standard modes (with or without interpolation clipping)")
    parser.add_argument("--num_points", type=int, default=192, help="Number of points for GP joint evaluation and distillation")
    parser.add_argument("--max_gamma", type=float, default=1.0, help="Max deformation intensity gamma when sample_mode is standard")
    parser.add_argument("--distill_target", type=str, default="sef", choices=["sef", "sef_stress", "sef_cauchy", "sef_split"], help="Target mode: strain energy function (sef), joint SEF + Piola stress (sef_stress), joint SEF + Cauchy stress (sef_cauchy), or separate DEV and VOL energy (sef_split)")
    parser.add_argument("--component", type=str, default="total", choices=["total", "dev", "vol"], help="Component to distill (used with sef_split)")
    parser.add_argument("--override_out_dir", type=str, default=None, help="Explicitly specify the output directory for distilled model logs and artifacts (overriding timestamp generation)")
    parser.add_argument("--load_existing_sensitivities", action="store_true", help="Skip Sobol resampling and directly load existing sensitivity CSVs from out_dir")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for PyTorch and Numpy")
    args = parser.parse_args()
    
    prefix = f"{args.component}_" if args.distill_target == "sef_split" else ""
    def pfx(filename): return prefix + filename

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    torch.set_default_dtype(torch.float64)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.sample_mode == "dataset_all":
        export_subfolder = "pytorch_export_dataset_all"
    elif args.sample_mode == "dataset_f":
        export_subfolder = f"pytorch_export_dataset_f_n{args.num_points}"
    elif args.sample_mode == "standard_interp":
        export_subfolder = "pytorch_export_standard_interp"
    elif args.sample_mode == "inducing_points":
        export_subfolder = "pytorch_export_inducing_points"
    else:
        export_subfolder = f"pytorch_export_standard_g{args.max_gamma}"
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
        import subprocess
        print(f"'{export_dir}' or '{mean_file}' not found. Exporting GP to PyTorch first (sample_mode: {args.sample_mode}, max_gamma: {args.max_gamma}, distill_target: {args.distill_target})...")
        subprocess.run(["python3", "distillation/export_gp_to_pytorch.py", "--saved_model_dir", args.saved_model_dir, "--sample_mode", args.sample_mode, "--num_points", str(args.num_points), "--max_gamma", str(args.max_gamma), "--distill_target", args.distill_target, "--export_subfolder", export_subfolder], check=True)
        
    mean_psi = torch.tensor(np.load(os.path.join(export_dir, mean_file)), dtype=torch.float64, device=device)
    cov_psi = torch.tensor(np.load(os.path.join(export_dir, cov_file)), dtype=torch.float64, device=device)
    cov_psi = (cov_psi + cov_psi.T) / 2.0
    
    # Ensure strict positive-definiteness without artificial diagonal inflation (eigenvalue clipping)
    L, V = torch.linalg.eigh(cov_psi)
    min_eig = L.min().item()
    if min_eig < 1e-8:
        print(f"Adjusting numerical eigenvalues (min eigen: {min_eig:.3e}) to ensure perfectly smooth, positive-definite paths...")
        L = torch.clamp(L, min=1e-8)
        cov_psi = V @ torch.diag(L) @ V.T
        cov_psi = (cov_psi + cov_psi.T) / 2.0
        try:
            torch.linalg.cholesky(cov_psi)
        except RuntimeError:
            print("Fallback: Cholesky failed after spectral clip. Adding trace jitter.")
            cov_psi += 1e-6 * torch.eye(cov_psi.shape[0], dtype=torch.float64, device=device)
    f3x3 = torch.tensor(np.load(os.path.join(export_dir, "f3x3.npy")), dtype=torch.float64, device=device)

    num_points = f3x3.shape[0]
    
    # Map deformation modes to test case identifiers for sensitivity analysis output grouping
    test_cases = torch.zeros(num_points, dtype=torch.int64, device=device)
    if args.sample_mode in ["dataset_f", "dataset_all"]:
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

    mock_gp = MockGP(mean_psi, cov_psi, device=device)
    include_log = args.material_model not in ["gmr_nolog", "gmr_no_log"]
    if args.component == "dev":
        model = PyTorchGMRDevModel(num_points, device=device, distill_target=args.distill_target, include_log_terms=include_log)
    elif args.component == "vol":
        model = PyTorchGMRVolModel(num_points, device=device, distill_target=args.distill_target)
    else:
        model = PyTorchGMRModel(num_points, device=device, distill_target=args.distill_target, include_log_terms=include_log)
    
    if args.distill_target == "sef_split":
        if args.component == "dev":
            model.deactivate_parameters([model.num_parameters - 3, model.num_parameters - 2, model.num_parameters - 1])
        elif args.component == "vol":
            model.deactivate_parameters(list(range(model.num_parameters - 3)))
            
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
        for p in ["isihara", "nh", "neohookean2", "nh2", "gentthomas", "nh4", "neohookean4", "c20d10d05", "c20_d10_d05"]:
            if p in parts:
                true_model_name = p
                break
                
        true_model = get_material(true_model_name, jit_P=False)
        
        true_params = {}
        if true_model_name == "isihara":
            true_params = {"$C_{10}$": true_model.c10, "$C_{01}$": true_model.c01, "$C_{20}$": true_model.c20, "$D_{1}$": true_model.d1}
        elif true_model_name in ["nh", "neohookean2", "nh2"]:
            true_params = {"$C_{10}$": true_model.dev_params[0], "$D_{1}$": true_model.vol_params[0]}
        elif true_model_name in ["nh4", "neohookean4"]:
            true_params = {"$C_{10}$": true_model.dev_params[0], "$D_{2}$": true_model.vol_params[1]}
        elif true_model_name in ["gentthomas"]:
            true_params = {"$C_{10}$": true_model.dev_params[0], "$E$": true_model.dev_params[9], "$D_{1}$": true_model.vol_params[0]}
        elif true_model_name in ["c20d10d05", "c20_d10_d05"]:
            true_params = {"$C_{10}$": true_model.dev_params[0], "$D_{1}$": true_model.vol_params[0], "$D_{2}$": true_model.vol_params[1]}
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
        if args.distill_target in ["sef_stress", "sef_cauchy", "sef_split"]:
            mode_str += f"_{args.distill_target}"
        if args.override_out_dir:
            out_dir = os.path.abspath(args.override_out_dir)
        else:
            out_dir = os.path.abspath(os.path.join("distillation", "distilled_models", f"{current_time}_{true_model_name}{noise_str}_{args.material_model}{mode_str}_uqmodeldisc"))
        log_mode = "w"
    os.makedirs(out_dir, exist_ok=True)
    
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
            samples_pre = distribution.sample(5000).cpu().numpy()
        full_param_names_pre = [n for i, n in enumerate(full_param_names_master) if model._parameter_mask[i].item()]
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
            axes_pre[i].set_title(f"{col} (Pre-Sensitivity)\nMean: {mean_val:.4f}", color='#2c3e50', fontsize=12, fontweight='bold')
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
                
                if args.pruning_mode == "ec":
                    denominator = max(np.sum(first_means), np.sum(tot_means))
                    if denominator == 0:
                        denominator = 1.0
                    
                    sorted_tot_means = tot_means[sorted_indices]
                    est_coverage_pct = np.cumsum(sorted_tot_means) / denominator * 100.0
                    
                    cutoff_idx = np.searchsorted(est_coverage_pct, args.ec_threshold)
                    if cutoff_idx >= len(est_coverage_pct):
                        cutoff_idx = len(est_coverage_pct) - 1
                        
                    selected_indices = sorted_indices[:cutoff_idx + 1].tolist()
                    print(f"\n[Pruning Mode: EC] Selecting top {cutoff_idx + 1} parameters to reach {args.ec_threshold}% Estimated Coverage.")
                
                elif args.pruning_mode == "threshold":
                    selected_indices = [i for i, mean in enumerate(tot_means) if mean >= args.sobol_threshold]
                    print(f"\n[Pruning Mode: Threshold] Selecting parameters with Total-Order > {args.sobol_threshold}.")
                
                model.deactivate_all_parameters()
                model.activate_parameters(selected_indices)
                
        except Exception as e:
            print(f"Error during custom parameter selection logic: {e}")
            print("Falling back to UQModelDisc default selection.")

        active_names = model.get_active_parameter_names()
        print(f"Selected sensitive parameters ({len(active_names)}): {active_names}")
        if len(active_names) == 0:
            print("Warning: No parameters exceeded Sobol threshold! Keeping all parameters active.")
            model.activate_parameters(list(range(model.num_parameters)))
        elif len(active_names) < model.num_parameters:
            print("\n========================================================================")
            model_type_str = f"({args.component.upper()} Split Model)" if args.distill_target == "sef_split" else "(Sparse Model)"
            print(f"=== Stage 2: Target Sub-Parameter Distillation {model_type_str} ===")
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
            
        samples_path = os.path.join(settings.PROJECT_DIR, pfx("flow_samples.npy"))
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
        plot_path = os.path.join(out_dir, pfx(f"distributions_{args.material_model}.pdf"))
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

    # Generate parameter violin plot
    try:
        print("Generating parameter violin plot...")
        plt.figure(figsize=(max(8, len(full_param_names_master) * 0.8), 6))
        
        violin_data = []
        violin_positions = []
        labels = []
        
        for i, col in enumerate(full_param_names_master):
            if col in model.parameter_names:
                violin_data.append(df[col].values)
                violin_positions.append(i)
            labels.append(col)
            
        ax = plt.gca()
        if violin_data:
            parts = ax.violinplot(violin_data, positions=violin_positions, showmeans=False, showextrema=False)
            for pc in parts['bodies']:
                pc.set_facecolor('#2980b9')
                pc.set_edgecolor('black')
                pc.set_alpha(0.7)
            
            for pos, data in zip(violin_positions, violin_data):
                mean_val = np.mean(data)
                ci_lower = np.percentile(data, 2.5)
                ci_upper = np.percentile(data, 97.5)
                
                ax.plot([pos - 0.2, pos + 0.2], [mean_val, mean_val], color='black', lw=2)
                ax.plot([pos, pos], [ci_lower, ci_upper], color='black', lw=2)
                
        ax.set_xticks(range(len(full_param_names_master)))
        ax.set_xticklabels(labels, fontsize=12)
        
        for i, col in enumerate(full_param_names_master):
            true_val = true_params.get(col, 0.0)
            ax.scatter([i], [true_val], color='red', marker='X', s=100, zorder=10, lw=2, label="Ground Truth" if i==0 else "")
            
        ax.set_ylim([0, 2.5])
        ax.set_ylabel("Material Parameter Value", fontsize=12)
        ax.set_title(f"Parameter Posterior Distribution ({args.material_model})", fontsize=14, fontweight='bold')
        
        handles, lbls = ax.get_legend_handles_labels()
        by_label = dict(zip(lbls, handles))
        if by_label:
            ax.legend(by_label.values(), by_label.keys(), loc='upper right')
            
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        violin_path = os.path.join(out_dir, pfx(f"parameter_violin_{args.material_model}.pdf"))
        plt.savefig(violin_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"Saved parameter violin plot to {violin_path}")
    except Exception as e:
        print(f"Error generating parameter violin plot: {e}")

    # Run validation plot scripts (only if not sef_split, otherwise we wait for both to finish in bash script)
    if args.distill_target != "sef_split":
        try:
            import subprocess
            subprocess.run(["python3", "plots/plot_distilled_validation.py", 
                            "--distilled_dir", out_dir, 
                            "--material_model", args.material_model,
                            "--distill_target", args.distill_target], check=True)
        except Exception as e:
            print(f"Error running validation plots: {e}")

    # Run deformation sensitivity plot script
    try:
        active_params_str = ",".join(model.parameter_names)
        import subprocess
        cmd = ["python3", "plots/plot_deformation_sensitivity.py", "--distilled_dir", out_dir, "--active_params", active_params_str]
        if args.distill_target == "sef_split":
            cmd.extend(["--component", args.component, "--distill_target", args.distill_target])
        subprocess.run(cmd, check=True)
    except Exception as e:
        print(f"Error running deformation sensitivity plots: {e}")

    # Run invariant sensitivity plot scripts
    try:
        active_params_str = ",".join(model.parameter_names)
        
        base_cmd = ["python3", "--distilled_dir", out_dir, "--active_params", active_params_str]
        if args.distill_target == "sef_split":
            base_cmd.extend(["--component", args.component, "--distill_target", args.distill_target])
            
        subprocess.run([base_cmd[0], "plots/plot_invariant_sensitivity.py"] + base_cmd[1:], check=True)
        subprocess.run([base_cmd[0], "plots/plot_invariant_sensitivity_3d_pairs.py"] + base_cmd[1:], check=True)
        subprocess.run([base_cmd[0], "plots/plot_invariant_sensitivity_3d_interactive.py"] + base_cmd[1:], check=True)
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
