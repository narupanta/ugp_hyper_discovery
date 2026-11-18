"""
Distillation models (PyTorch): the candidate strain-energy models whose parameter distribution is distilled from the
GP teacher (uqmodeldisc's Wasserstein distillation of a normalizing flow), and the teacher's function-space samples.

  MockGP / CachedCholeskyDistribution:  GP posterior over the export states, f = m + L z (L = cached Cholesky factor)
  EnergyOutputSelector:                 all GP outputs are energies (no output selection)
  BasePyTorchMaterialModel:             parameter masking / population bookkeeping of uqmodeldisc
  PyTorchGMRModel                       generalized Mooney-Rivlin energy (dev + vol), joint distillation
  PyTorchGMRDevModel                    deviatoric GMR library (+ Ogden terms, core.material_library)
  PyTorchGMRVolModel                    volumetric library
  PyTorchGMRAnisoModel / ...CompModel   fibre terms (one or two fibre families)

Used by pipeline/distill.py.
"""
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "UQInModelDiscovery")))
from uqmodeldisc.models.base import (  # noqa: E402
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
from core.material_library import N_OGDEN_TERMS, OGDEN_ALPHA_INIT  # noqa: E402

OGDEN_PARAM_NAMES = tuple(n for k in range(1, N_OGDEN_TERMS + 1) for n in (f"$\\mu_{{{k}}}$", f"$\\alpha_{{{k}}}$"))
OGDEN_MU_NAMES, OGDEN_ALPHA_NAMES = OGDEN_PARAM_NAMES[0::2], OGDEN_PARAM_NAMES[1::2]


class CachedCholeskyDistribution:
    """
    Lightweight Gaussian process realization distribution using precomputed / cached
    lower-triangular Cholesky factor L:
        f = m + z @ L.T,  where z ~ N(0, I)
    Avoids torch.distributions.MultivariateNormal overhead, achieving a 3.5x - 5x speedup
    during GP functional draws in Wasserstein distillation.
    """
    def __init__(self, loc: torch.Tensor, scale_tril: torch.Tensor):
        self.loc = loc
        self.scale_tril = scale_tril

    @property
    def mean(self):
        return self.loc

    @property
    def covariance_matrix(self):
        return self.scale_tril @ self.scale_tril.T

    def rsample(self, sample_shape=torch.Size()):
        if isinstance(sample_shape, int):
            shape = (sample_shape, len(self.loc))
        elif isinstance(sample_shape, (torch.Size, tuple, list)):
            shape = tuple(sample_shape) + (len(self.loc),)
        else:
            shape = (len(self.loc),)
        z = torch.randn(shape, dtype=self.loc.dtype, device=self.loc.device)
        return self.loc + z @ self.scale_tril.T

    def sample(self, sample_shape=torch.Size()):
        return self.rsample(sample_shape)


class MockGP(nn.Module):
    def __init__(self, loc, cov, device, sampling_mode="cached_L", L=None):
        super().__init__()
        self.loc = loc.to(device)
        self.cov = cov.to(device)
        self.sampling_mode = sampling_mode
        self.device = device
        self.likelihood = nn.Module()
        if self.sampling_mode == "cached_L":
            if L is not None:
                self.scale_tril = L.to(device)
            else:
                self.scale_tril = torch.linalg.cholesky(self.cov)
        elif self.sampling_mode == "mvn":
            self.dist = torch.distributions.MultivariateNormal(self.loc, self.cov)

    def forward(self, inputs):
        if self.sampling_mode == "cached_L":
            return CachedCholeskyDistribution(self.loc, self.scale_tril)
        elif self.sampling_mode == "mvn":
            return self.dist
        else:
            raise ValueError(f"Unsupported sampling_mode: '{self.sampling_mode}' (choose 'cached_L' or 'mvn')")

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
    """
    Deviatoric candidate library. library = "gmr": polynomial terms in (I1_bar - 3, I2_bar - 3) (+ E log(I2_bar/3));
    "ogden": N_OGDEN_TERMS (8) Ogden terms 2 mu_k / alpha_k^2 (sum_i lam_bar_i^alpha_k - 3) (standard convention,
    core/ogden.py; parameters mu_1, alpha_1, ..., alpha_8); "gmr_ogden": both, polynomial terms first. All flow outputs
    are positive (exponential output of the flow): the coefficients and mu_k > 0 (mu_k is the term's shear modulus
    contribution, so every term is stable). The exponent is real and measured from the term's start exponent
    alpha0_k (OGDEN_ALPHA_INIT = -4, -3, -2, -1, 1, 2, 3, 4): alpha_k = alpha0_k + ln a_k with the flow's positive output
    a_k (= exp(z), so the shift z takes either sign and a fresh flow starts around alpha0_k); an inactive a_k (masked to
    0) gives alpha_k = alpha0_k exactly, which is how stage 1 fits the mu_k at the fixed exponents. Saved samples hold
    the physical (mu_k, alpha_k) (physical_samples).
    """
    def __init__(self, num_points, device, include_log_terms=False, distill_target="sef", library="gmr"):
        super().__init__()
        self.include_log_terms = include_log_terms
        self.distill_target = distill_target
        self.library = library
        self.use_poly = library in ("gmr", "gmr_ogden")
        self.use_ogden = library in ("ogden", "gmr_ogden")
        if self.use_ogden and distill_target in ["sef_stress", "sef_cauchy"]:
            raise NotImplementedError("Ogden libraries support energy targets (sef, sef_split) only")
        self._output_dim = 4 if distill_target in ["sef_stress", "sef_cauchy"] else 1

        poly_names = ("$C_{10}$", "$C_{01}$", "$C_{20}$", "$C_{11}$", "$C_{02}$", "$C_{30}$", "$C_{21}$", "$C_{12}$", "$C_{03}$")
        if self.include_log_terms:
            poly_names = poly_names + ("$E$",)
        names = (poly_names if self.use_poly else ()) + (OGDEN_PARAM_NAMES if self.use_ogden else ())
        self._num_poly = len(poly_names) if self.use_poly else 0
        self._num_parameters = len(names)
        self._parameter_names = names
        
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
    @staticmethod
    def physical_samples(samples, names):
        """Flow samples (mu_k > 0, a_k >= 0) -> physical (mu_k, alpha_k = alpha0_k + ln a_k; a_k = 0 -> alpha0_k)."""
        out = np.array(samples, dtype=float, copy=True)
        names = list(names)
        for k in range(1, N_OGDEN_TERMS + 1):
            al_n = f"$\\alpha_{{{k}}}$"
            if al_n not in names:
                continue
            a = out[:, names.index(al_n)]
            alpha = OGDEN_ALPHA_INIT[k - 1] + np.where(a > 0, np.log(np.where(a > 0, a, 1.0)), 0.0)
            out[:, names.index(al_n)] = alpha
        return out

    def __call__(self, inputs, test_cases, parameters, validate_args=False):
        full_parameters = mask_and_populate_parameters(parameters, self._parameter_mask, self._parameter_population_matrix)
        W_ogden = 0.0
        if self.use_ogden:
            from core.ogden import ogden_dev_energy_torch
            og = full_parameters[self._num_poly:]
            a = og[1::2]
            alpha0 = torch.as_tensor(OGDEN_ALPHA_INIT[:a.shape[0]], dtype=a.dtype, device=a.device)
            alphas = alpha0 + torch.where(a > 0, torch.log(torch.where(a > 0, a, torch.ones_like(a))), torch.zeros_like(a))
            W_ogden = ogden_dev_energy_torch(inputs, og[0::2], alphas)
            if not self.use_poly:
                return W_ogden.unsqueeze(-1)
            full_parameters = full_parameters[:self._num_poly]
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
        energy = W_dev + W_ogden

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


class PyTorchGMRAnisoCompModel(BasePyTorchMaterialModel):
    def __init__(self, num_points, device, theta1=np.pi/4.0, theta2=-np.pi/4.0, distill_target="sef"):
        super().__init__()
        self.distill_target = distill_target
        self._output_dim = 4 if distill_target in ["sef_stress", "sef_cauchy"] else 1
        
        self._num_parameters = 6
        self._parameter_names = ("$C_{42}$", "$C_{43}$", "$C_{44}$", "$C_{62}$", "$C_{63}$", "$C_{64}$")
        
        self._parameter_scales = torch.ones(self._num_parameters, device=device)
        self._device = device
        self._num_points = num_points
        self._parameter_mask = init_parameter_mask(self._num_parameters, device)
        self._parameter_population_matrix = init_parameter_population_matrix(self._num_parameters, device)
        self.a1 = torch.tensor([np.cos(theta1), np.sin(theta1), 0.0], dtype=torch.float64, device=device)
        self.a2 = torch.tensor([np.cos(theta2), np.sin(theta2), 0.0], dtype=torch.float64, device=device)

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
        J = torch.det(inputs)
        C_bar = (J**(-2/3)).unsqueeze(-1).unsqueeze(-1) * C
        
        I4_bar_1 = torch.einsum('i,...ij,j->...', self.a1, C_bar, self.a1)
        I4_bar_2 = torch.einsum('i,...ij,j->...', self.a2, C_bar, self.a2)
        
        I4_m1 = I4_bar_1 - 1.0
        I6_m1 = I4_bar_2 - 1.0

        C42, C43, C44, C62, C63, C64 = full_parameters

        W_aniso = (C42 * I4_m1**2 + C43 * I4_m1**3 + C44 * I4_m1**4 +
                   C62 * I6_m1**2 + C63 * I6_m1**3 + C64 * I6_m1**4)
        energy = W_aniso

        if self.distill_target in ["sef_stress", "sef_cauchy"]:
            F_inv_T = torch.linalg.inv(inputs).transpose(1, 2)
            dI4_1_dF = 2.0 * (J.view(-1, 1, 1)**(-2/3)) * torch.einsum('i,...j->...ij', self.a1, torch.matmul(inputs, self.a1)) - (2.0/3.0) * I4_bar_1.view(-1, 1, 1) * F_inv_T
            dI4_2_dF = 2.0 * (J.view(-1, 1, 1)**(-2/3)) * torch.einsum('i,...j->...ij', self.a2, torch.matmul(inputs, self.a2)) - (2.0/3.0) * I4_bar_2.view(-1, 1, 1) * F_inv_T
            
            dW_dI4_1 = 2.0 * C42 * I4_m1 + 3.0 * C43 * I4_m1**2 + 4.0 * C44 * I4_m1**3
            dW_dI4_2 = 2.0 * C62 * I6_m1 + 3.0 * C63 * I6_m1**2 + 4.0 * C64 * I6_m1**3
            
            P = dW_dI4_1.view(-1, 1, 1) * dI4_1_dF + dW_dI4_2.view(-1, 1, 1) * dI4_2_dF
            if self.distill_target == "sef_cauchy":
                cauchy = (1.0 / J.view(-1, 1, 1)) * torch.matmul(P, inputs.transpose(1, 2))
                return torch.cat([energy.unsqueeze(1), cauchy[:, 0, 0].unsqueeze(1), cauchy[:, 1, 1].unsqueeze(1), cauchy[:, 0, 1].unsqueeze(1)], dim=1)
            else:
                return torch.cat([energy.unsqueeze(1), P[:, 0, 0].unsqueeze(1), P[:, 1, 1].unsqueeze(1), P[:, 0, 1].unsqueeze(1)], dim=1)
        return energy.unsqueeze(-1)


class PyTorchGMRAnisoModel(BasePyTorchMaterialModel):
    def __init__(self, num_points, device, theta1=np.pi/4.0, theta2=-np.pi/4.0, distill_target="sef"):
        super().__init__()
        self.distill_target = distill_target
        self._output_dim = 4 if distill_target in ["sef_stress", "sef_cauchy"] else 1
        
        self._num_parameters = 18
        self._parameter_names = (
            "$C_{10}$", "$C_{01}$", "$C_{20}$", "$C_{11}$", "$C_{02}$", "$C_{30}$", "$C_{21}$", "$C_{12}$", "$C_{03}$",
            "$D_{1}$", "$D_{2}$", "$D_{3}$",
            "$C_{42}$", "$C_{43}$", "$C_{44}$", "$C_{62}$", "$C_{63}$", "$C_{64}$"
        )
        
        self._parameter_scales = torch.ones(self._num_parameters, device=device)
        self._device = device
        self._num_points = num_points
        self._parameter_mask = init_parameter_mask(self._num_parameters, device)
        self._parameter_population_matrix = init_parameter_population_matrix(self._num_parameters, device)
        self.a1 = torch.tensor([np.cos(theta1), np.sin(theta1), 0.0], dtype=torch.float64, device=device)
        self.a2 = torch.tensor([np.cos(theta2), np.sin(theta2), 0.0], dtype=torch.float64, device=device)

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

        C_bar = (J**(-2/3)).unsqueeze(-1).unsqueeze(-1) * C
        I4_bar_1 = torch.einsum('i,...ij,j->...', self.a1, C_bar, self.a1)
        I4_bar_2 = torch.einsum('i,...ij,j->...', self.a2, C_bar, self.a2)
        I4_m1 = I4_bar_1 - 1.0
        I6_m1 = I4_bar_2 - 1.0

        C10, C01, C20, C11, C02, C30, C21, C12, C03, D1, D2, D3, C42, C43, C44, C62, C63, C64 = full_parameters

        W_dev = (C10 * I1_m3 + C01 * I2_m3 + C20 * I1_m3**2 + C11 * I1_m3 * I2_m3 + C02 * I2_m3**2 +
                 C30 * I1_m3**3 + C21 * (I1_m3**2) * I2_m3 + C12 * I1_m3 * (I2_m3**2) + C03 * I2_m3**3)
        W_vol = D1 * J_m1**2 + D2 * J_m1**4 + D3 * J_m1**6
        W_aniso = (C42 * I4_m1**2 + C43 * I4_m1**3 + C44 * I4_m1**4 +
                   C62 * I6_m1**2 + C63 * I6_m1**3 + C64 * I6_m1**4)

        energy = W_dev + W_vol + W_aniso

        if self.distill_target in ["sef_stress", "sef_cauchy"]:
            F_inv_T = torch.linalg.inv(inputs).transpose(1, 2)
            dJ_dF = J.view(-1, 1, 1) * F_inv_T
            dI1bar_dF = J.view(-1, 1, 1)**(-2/3) * (2.0 * inputs - (2.0/3.0) * I1.view(-1, 1, 1) * F_inv_T)
            dI2_dF = 2.0 * I1.view(-1, 1, 1) * inputs - 2.0 * torch.matmul(inputs, torch.matmul(inputs.transpose(1, 2), inputs))
            dI2bar_dF = J.view(-1, 1, 1)**(-4/3) * (dI2_dF - (4.0/3.0) * I2.view(-1, 1, 1) * F_inv_T)
            
            dI4_1_dF = 2.0 * (J.view(-1, 1, 1)**(-2/3)) * torch.einsum('i,...j->...ij', self.a1, torch.matmul(inputs, self.a1)) - (2.0/3.0) * I4_bar_1.view(-1, 1, 1) * F_inv_T
            dI4_2_dF = 2.0 * (J.view(-1, 1, 1)**(-2/3)) * torch.einsum('i,...j->...ij', self.a2, torch.matmul(inputs, self.a2)) - (2.0/3.0) * I4_bar_2.view(-1, 1, 1) * F_inv_T

            dW_dI1 = C10 + 2.0 * C20 * I1_m3 + C11 * I2_m3 + 3.0 * C30 * (I1_m3**2) + 2.0 * C21 * I1_m3 * I2_m3 + C12 * (I2_m3**2)
            dW_dI2 = C01 + C11 * I1_m3 + 2.0 * C02 * I2_m3 + C21 * (I1_m3**2) + 2.0 * C12 * I1_m3 * I2_m3 + 3.0 * C03 * (I2_m3**2)
            dW_dJ = 2.0 * D1 * J_m1 + 4.0 * D2 * J_m1**3 + 6.0 * D3 * J_m1**5
            dW_dI4_1 = 2.0 * C42 * I4_m1 + 3.0 * C43 * I4_m1**2 + 4.0 * C44 * I4_m1**3
            dW_dI4_2 = 2.0 * C62 * I6_m1 + 3.0 * C63 * I6_m1**2 + 4.0 * C64 * I6_m1**3

            P = dW_dI1.view(-1, 1, 1) * dI1bar_dF + dW_dI2.view(-1, 1, 1) * dI2bar_dF + dW_dJ.view(-1, 1, 1) * dJ_dF + dW_dI4_1.view(-1, 1, 1) * dI4_1_dF + dW_dI4_2.view(-1, 1, 1) * dI4_2_dF
            
            if self.distill_target == "sef_cauchy":
                cauchy = (1.0 / J.view(-1, 1, 1)) * torch.matmul(P, inputs.transpose(1, 2))
                return torch.cat([energy.unsqueeze(1), cauchy[:, 0, 0].unsqueeze(1), cauchy[:, 1, 1].unsqueeze(1), cauchy[:, 0, 1].unsqueeze(1)], dim=1)
            else:
                return torch.cat([energy.unsqueeze(1), P[:, 0, 0].unsqueeze(1), P[:, 1, 1].unsqueeze(1), P[:, 0, 1].unsqueeze(1)], dim=1)
        return energy.unsqueeze(-1)
