import os

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
jax.config.update("jax_enable_x64", True)
from .material_models import get_material
from .utils import (
    B_func,
    I1_func,
    I2_func,
    I3_func,
    J_func,
    solve_for_coefficients_batched,
    deformation_gradient_element
)

from abc import ABC, abstractmethod

class HyperelasticDataset(ABC):
    @abstractmethod
    def get_data(self):
        pass

class PrecomputedVFMDataset(HyperelasticDataset):
    def __init__(self, data_path: os.PathLike):
        self.data_path = data_path
        
    def get_data(self):
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Dataset not found at {self.data_path}")
        return dict(np.load(self.data_path, allow_pickle=True))

class DatasetFactory:
    @staticmethod
    def create(dataset_type: str, **kwargs) -> HyperelasticDataset:
        if dataset_type == "dataset/precomputed_vfm":
            return PrecomputedVFMDataset(kwargs["data_path"])
        elif dataset_type == "benchmark":
            return BenchmarkDataset(kwargs["data_dir"], kwargs["noise"], kwargs["mat_model"])
        elif dataset_type == "traction":
            return TractionDataset(kwargs.get("data_dir", "/home/mmdiscovery/shared/dataset/isihara_fix"))
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")

class BenchmarkDataset(HyperelasticDataset):
    def __init__(self, data_dir: os.PathLike, noise: str, mat_model: str):
        self.data_dir = data_dir
        self.noise = noise
        self.mat_model = mat_model
        self.mat_model_path = os.path.join(data_dir, noise, mat_model)
        self.loadsteps = os.listdir(self.mat_model_path)

    def __len__(self):

        return len(self.loadsteps)
    
    def __getitem__(self, loadstep):
        files_path = os.path.join(self.mat_model_path, str(loadstep))
        files = os.listdir(files_path)
        data = dict()
        for f in files :
            if f.endswith(".csv") :
                data[f"{f.split(".")[0]}"] = pd.read_csv(files_path + "/" + f)
        output_nodes = data["output_nodes"]
        output_elements = data["output_elements"]
        output_reactions = data["output_reactions"]
        output_integrator = data["output_integrator"]
        if 'ux_orig' in output_nodes.columns and 'uy_orig' in output_nodes.columns:
            output_nodes.ux[output_nodes.bcx!=0] = output_nodes.ux_orig[output_nodes.bcx!=0]
            output_nodes.uy[output_nodes.bcy!=0] = output_nodes.uy_orig[output_nodes.bcy!=0]
        mesh_pos = output_nodes[["x", "y"]].to_numpy()
        # u = output_nodes[["ux_orig", "uy_orig"]].to_numpy()

        u = output_nodes[["ux", "uy"]].to_numpy()
        bc = output_nodes[["bcx", "bcy"]].to_numpy()

        cells = output_elements[["node1", "node2", "node3"]].to_numpy()
        P2 = output_elements[["Pxx", "Pxy", "Pyx", "Pyy"]].to_numpy()
        F2 = output_elements[["Fxx", "Fxy", "Fyx", "Fyy"]].to_numpy()

        coords_elems = mesh_pos[cells]
        disp_elems   = u[cells]

        # Vectorize the function
        f = jax.vmap(lambda ce, de: deformation_gradient_element(ce, de))(coords_elems, disp_elems)


                # Reshape to (n, 2, 2)
        # --- 1. reshape ---
        P2 = P2.reshape(-1, 2, 2)
        F2 = F2.reshape(-1, 2, 2)

        # --- 2. create output arrays ---
        P = jnp.zeros((P2.shape[0], 3, 3))
        F = jnp.tile(jnp.eye(3), (P2.shape[0], 1, 1))

        # --- 3. JAX-style assignment ---
        P = P.at[:, :2, :2].set(P2)
        F = F.at[:, :2, :2].set(F2)
        mm = get_material(self.mat_model.lower())
        P_from_mm = mm.P(F)
        J = J_func(F)

        sigma = 1/J[:, None, None] * P_from_mm @ jnp.swapaxes(F, -2, -1)
        B_train = B_func(F)
        I1_train = I1_func(B_train)
        I2_train = I2_func(B_train)
        I3_train = I3_func(B_train)
        invariants = jnp.stack([I1_train, I2_train, I3_train], axis=-1)

        # Eigenvalues
        B_eig_val = jnp.real(jnp.linalg.eigvalsh(B_train))
        sigma_eig_val = jnp.real(jnp.linalg.eigvalsh(sigma))

        coeffs, _ = solve_for_coefficients_batched(B_eig_val, sigma_eig_val)

        reaction_forces = output_reactions["forces"] #right, left, top, bottom 
        data = dict(F = F, P = P_from_mm, sigma = sigma, coeffs = coeffs, invariants = invariants, 
                    cells = cells, coords_elems = coords_elems, disp_elems = disp_elems, bc = bc, reaction_forces = reaction_forces)
        return data
        
    def get_data(self):
        # Implementation for the full dataset retrieval if needed
        return [self[i] for i in range(len(self))]

class TractionDataset(HyperelasticDataset):
    def __init__(self, data_dir: os.PathLike = "/home/mmdiscovery/shared/dataset/isihara_fix"):
        self.data_dir = data_dir
        self.files = os.listdir(self.data_dir)
    def __len__(self) :
        return len(self.files)
    def __getitem__(self, idx) :
        data = np.load(os.path.join(self.data_dir, self.files[idx]))
        return data

    def get_data(self):
        return [self[i] for i in range(len(self))]

        
# Example usage
if __name__ == "__main__":
    dataset = BenchmarkDataset("dataset/benchmarks", "noise=low", "Isihara")
    loadsteps = [10, 50, 80]
    check = dataset[10]
    data = TestSpecimen("dataset/benchmarks/test-specimen", "Isihara-GT")
    check = data[10]
    print('')
    # dataset = UniaxialGenerator(n_samples=10, gamma_range=(0.5, 2.0), mat_model="MooneyRivlin")
    # F = dataset.get_F()
    # invariants = dataset.get_invariants(F)
    # sigma = dataset.get_cauchy_stress(F)
    # coeffs = dataset.get_coeffs(F)

    # print("F shape:", F.shape)
    # print("Invariants shape:", invariants.shape)
    # print("Cauchy stress shape:", sigma.shape)
    # print("Coefficients shape:", coeffs.shape)

