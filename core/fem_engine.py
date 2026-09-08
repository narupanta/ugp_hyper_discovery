"""
fem_engine.py (Universal, Modular FEM Engine for Hyperelasticity)

Architecture & Features:
1. Modular Geometry Subsystem (BaseGeometry, BlockGeometry, HolesGeometry)
   - Parametric Gmsh mesh generation and export to .npz format.
   - Spatial boundary predicates for named surface boundaries.
2. Modular Boundary Condition Configuration:
   - DirichletBC: Explicit DOF constraints and prescribed displacement support.
   - NeumannBC: Surface traction direction vectors and load schedule indexing.
   - Standard 5-channel node_type generator:
     [is_internal, is_fix_x, is_fix_y, is_traction_x, is_traction_y]
3. Universal JAX-FEM HyperElasticity Problem & Adaptive Solver:
   - HyperElasticityProblem: supports arbitrary stress functions P(F) or internal variable parameters.
   - solve_adaptive_fem: robust incremental solver with automatic step-halving for convergence.
4. Complete Kinematics & Invariants Computation:
   - Isochoric: I1_bar, I2_bar, J.
   - Anisotropic: I4_bar, I6_bar, I8_bar (omitted when isotropic).
5. Comprehensive Dataset Exporter:
   - Ground truth fields: u_true, F_true, true_I1_bar, true_I2_bar, true_J (and true_I4_bar, true_I6_bar, true_I8_bar if anisotropic).
   - Observed fields with noise: u_obs (and u), F_obs (and F), obs_I1_bar, obs_I2_bar, obs_J.
"""

import os
import sys
import math
from pathlib import Path
from typing import Dict, List, Tuple, Callable, Optional, Union, Any

import numpy as np
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

import gmsh
from mpi4py import MPI
from dolfinx.io.gmsh import read_from_msh
from jax_fem.problem import Problem
from jax_fem.solver import solver
from jax_fem.generate_mesh import get_meshio_cell_type, Mesh

from core.utils import (
    fto3x3,
    deformation_gradient_element,
    transformation_jacobian,
    compute_invariants_np,
)
from core.loss_function import neumann_cell_force
from core.material_models import get_material

try:
    from dataset.convert_msh_to_npz import convert_msh_to_npz
except ImportError:
    from convert_msh_to_npz import convert_msh_to_npz


# ==============================================================================
# 1. Modular Geometry & Mesh Generators
# ==============================================================================

class BaseGeometry:
    """Abstract base geometry defining mesh generation and boundary predicates."""
    def generate_mesh(self, msh_path: str, npz_path: str) -> None:
        raise NotImplementedError

    def get_boundary_predicates(self) -> Dict[str, Callable[[jnp.ndarray], jnp.ndarray]]:
        """
        Returns a dictionary mapping boundary names ('left', 'bottom', 'right', 'top')
        to point filter callables point -> bool.
        """
        return {
            "left": lambda pt: jnp.isclose(pt[0], 0.0, atol=1e-6),
            "bottom": lambda pt: jnp.isclose(pt[1], 0.0, atol=1e-6),
            "right": lambda pt: jnp.isclose(pt[0], 1.0, atol=1e-6),
            "top": lambda pt: jnp.isclose(pt[1], 1.0, atol=1e-6),
        }


class BlockGeometry(BaseGeometry):
    """Square domain with a quarter-circular cutout at (0, 0)."""
    def __init__(self, Lx: float = 1.0, Ly: float = 1.0, R_hole: float = 0.1, mesh_size: float = 0.08):
        self.Lx = Lx
        self.Ly = Ly
        self.R_hole = R_hole
        self.mesh_size_far = mesh_size
        self.mesh_size_near = mesh_size / 4.0

    def generate_mesh(self, msh_path: str, npz_path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(msh_path)), exist_ok=True)
        gmsh.initialize()
        model = gmsh.model.occ

        rect = model.addRectangle(0.0, 0.0, 0.0, self.Lx, self.Ly)
        circle = model.addDisk(0.0, 0.0, 0.0, self.R_hole, self.R_hole)
        out_tags, _ = model.cut([(2, rect)], [(2, circle)])
        model.synchronize()

        all_curves = gmsh.model.getEntities(1)
        hole_curve_tag = []
        for dim, tag in all_curves:
            min_x, min_y, _, max_x, max_y, _ = gmsh.model.getBoundingBox(dim, tag)
            if max_x <= self.R_hole + 1e-6 and min_x >= -self.R_hole - 1e-6:
                if max_y <= self.R_hole + 1e-6 and min_y >= -self.R_hole - 1e-6:
                    hole_curve_tag.append(tag)

        gmsh.model.mesh.field.add("Distance", 1)
        gmsh.model.mesh.field.setNumbers(1, "CurvesList", hole_curve_tag)

        gmsh.model.mesh.field.add("Threshold", 2)
        gmsh.model.mesh.field.setNumber(2, "InField", 1)
        gmsh.model.mesh.field.setNumber(2, "SizeMin", self.mesh_size_near)
        gmsh.model.mesh.field.setNumber(2, "SizeMax", self.mesh_size_far)
        gmsh.model.mesh.field.setNumber(2, "DistMin", 0.02)
        gmsh.model.mesh.field.setNumber(2, "DistMax", 0.36)
        gmsh.model.mesh.field.setAsBackgroundMesh(2)

        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)

        surf_tag = out_tags[0][1]
        gmsh.model.addPhysicalGroup(2, [surf_tag], 1, name="domain")
        gmsh.model.mesh.generate(2)
        gmsh.write(msh_path)
        gmsh.finalize()

        _ = read_from_msh(msh_path, MPI.COMM_WORLD, 0, 2)
        convert_msh_to_npz(msh_path, npz_path)


class HolesGeometry(BaseGeometry):
    """Domain with two asymmetric elliptical cutouts."""
    def __init__(self, Lx: float = 1.0, Ly: float = 1.0, mesh_size: float = 0.08):
        self.Lx = Lx
        self.Ly = Ly
        self.mesh_size_far = mesh_size
        self.mesh_size_near = mesh_size * 0.3

    def generate_mesh(self, msh_path: str, npz_path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(msh_path)), exist_ok=True)
        gmsh.initialize()
        model = gmsh.model.occ

        rect = model.addRectangle(0.0, 0.0, 0.0, self.Lx, self.Ly)
        hole1 = model.addDisk(0.7, 0.7, 0.0, 0.2, 0.18)
        hole2 = model.addDisk(0.25, 0.3, 0.0, 0.18, 0.04)
        model.rotate([(2, hole2)], 0.25, 0.3, 0.0, 0.0, 0.0, 1.0, math.pi / 2.0)

        out_tags, _ = model.cut([(2, rect)], [(2, hole1), (2, hole2)])
        model.synchronize()

        all_curves = gmsh.model.getEntities(1)
        hole_curve_tag = []
        for dim, tag in all_curves:
            min_x, min_y, _, max_x, max_y, _ = gmsh.model.getBoundingBox(dim, tag)
            if (min_x > 0.01 and max_x < 0.99 and min_y > 0.01 and max_y < 0.99):
                hole_curve_tag.append(tag)

        gmsh.model.mesh.field.add("Distance", 1)
        gmsh.model.mesh.field.setNumbers(1, "CurvesList", hole_curve_tag)

        gmsh.model.mesh.field.add("Threshold", 2)
        gmsh.model.mesh.field.setNumber(2, "InField", 1)
        gmsh.model.mesh.field.setNumber(2, "SizeMin", self.mesh_size_near)
        gmsh.model.mesh.field.setNumber(2, "SizeMax", self.mesh_size_far)
        gmsh.model.mesh.field.setNumber(2, "DistMin", 0.02)
        gmsh.model.mesh.field.setNumber(2, "DistMax", 0.3)
        gmsh.model.mesh.field.setAsBackgroundMesh(2)

        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)

        surf_tag = out_tags[0][1]
        gmsh.model.addPhysicalGroup(2, [surf_tag], 1, name="domain")
        gmsh.model.mesh.generate(2)
        gmsh.write(msh_path)
        gmsh.finalize()

        _ = read_from_msh(msh_path, MPI.COMM_WORLD, 0, 2)
        convert_msh_to_npz(msh_path, npz_path)


_GEOMETRY_REGISTRY: Dict[str, Callable[..., BaseGeometry]] = {
    "block": BlockGeometry,
    "holes": HolesGeometry,
}

def get_geometry(name: str, **kwargs) -> BaseGeometry:
    name_clean = name.lower()
    if name_clean not in _GEOMETRY_REGISTRY:
        raise ValueError(f"Unknown geometry '{name}'. Available: {list(_GEOMETRY_REGISTRY.keys())}")
    return _GEOMETRY_REGISTRY[name_clean](**kwargs)


# ==============================================================================
# 2. Modular Boundary Condition Specifications
# ==============================================================================

class DirichletBC:
    """
    Specifies a Dirichlet boundary condition on a boundary.
    dof: 0 for u_x, 1 for u_y
    val: float or callable pt -> float
    is_prescribed: True if this DOF carries prescribed displacement increments.
    """
    def __init__(self, name: str, location_fn: Callable, dof: int, 
                 val: Union[float, Callable] = 0.0, is_prescribed: bool = False):
        self.name = name
        self.location_fn = location_fn
        self.dof = int(dof)
        self.val = val
        self.is_prescribed = is_prescribed


class NeumannBC:
    """
    Specifies a Neumann traction boundary condition on a boundary.
    direction: vector (tx, ty) indicating traction direction.
               In JAX-FEM, tensile traction on outward normal n=[1,0] corresponds to direction=[-1.0, 0.0].
    load_index: index in the load schedule vector (e.g. 0 for right, 1 for top).
    """
    def __init__(self, name: str, location_fn: Callable, direction: Tuple[float, float], load_index: int = 0):
        self.name = name
        self.location_fn = location_fn
        self.direction = tuple(direction)
        self.load_index = int(load_index)


class BoundaryConditionConfig:
    """
    Universal boundary condition container supporting both force and displacement control modes.
    Encapsulates Dirichlet and Neumann constraints without hardcoding geometric boundaries.
    """
    def __init__(
        self,
        mode: str = "force",
        dirichlet_bcs: Optional[List[DirichletBC]] = None,
        neumann_bcs: Optional[List[NeumannBC]] = None
    ):
        self.mode = mode.lower()
        self.dirichlet_bcs = dirichlet_bcs or []
        self.neumann_bcs = neumann_bcs or []

    def get_dirichlet_info(self):
        """
        Constructs Dirichlet BC triplet required by JAX-FEM:
        [ [location_fn1, ...], [dof1, ...], [val_fn1, ...] ]
        """
        location_fns = [bc.location_fn for bc in self.dirichlet_bcs]
        dofs = [bc.dof for bc in self.dirichlet_bcs]
        val_fns = []
        for bc in self.dirichlet_bcs:
            if callable(bc.val):
                val_fns.append(bc.val)
            else:
                c_val = float(bc.val)
                val_fns.append(lambda pt, v=c_val: v)
        return [location_fns, dofs, val_fns]

    def get_surface_maps(self) -> List[Callable]:
        """
        Constructs JAX-FEM surface traction map callables for all Neumann boundaries.
        """
        surface_maps = []
        for nbc in self.neumann_bcs:
            dir_vec = jnp.array(nbc.direction, dtype=jnp.float64)
            def make_surface_map(d=dir_vec):
                def surface_map(u, x, load_val):
                    return d * load_val[0]
                return surface_map
            surface_maps.append(make_surface_map())
        return surface_maps

    def create_node_type_array(self, node_coords: jnp.ndarray) -> jnp.ndarray:
        """
        Constructs the standard 5-channel one-hot node_type array:
        [is_internal, is_fix_x, is_fix_y, is_traction_x, is_traction_y]
        Completely decoupled from boundary names: checks Dirichlet DOFs and Neumann traction directions.
        """
        n_nodes = node_coords.shape[0]
        is_fix_x = jnp.zeros(n_nodes, dtype=bool)
        is_fix_y = jnp.zeros(n_nodes, dtype=bool)
        is_traction_x = jnp.zeros(n_nodes, dtype=bool)
        is_traction_y = jnp.zeros(n_nodes, dtype=bool)

        # 1. Evaluate Dirichlet DOFs
        for dbc in self.dirichlet_bcs:
            mask = jax.vmap(dbc.location_fn)(node_coords)
            if dbc.dof == 0:
                is_fix_x = is_fix_x | mask
            elif dbc.dof == 1:
                is_fix_y = is_fix_y | mask

        # 2. Evaluate Neumann Traction DOFs
        for nbc in self.neumann_bcs:
            mask = jax.vmap(nbc.location_fn)(node_coords)
            if abs(nbc.direction[0]) > 1e-6:
                is_traction_x = is_traction_x | mask
            if abs(nbc.direction[1]) > 1e-6:
                is_traction_y = is_traction_y | mask

        is_internal = ~(is_fix_x | is_fix_y | is_traction_x | is_traction_y)
        node_type = jnp.stack([
            is_internal, is_fix_x, is_fix_y, is_traction_x, is_traction_y
        ], axis=-1).astype(jnp.float32)
        return node_type


def create_default_bc_config(
    geometry_name: str = "block",
    mode: str = "force",
    pred_dict: Optional[Dict[str, Callable]] = None
) -> BoundaryConditionConfig:
    """
    Factory helper creating standard boundary conditions for standard geometries.
    """
    if pred_dict is None:
        geom = get_geometry(geometry_name)
        pred_dict = geom.get_boundary_predicates()

    mode = mode.lower()
    geometry_name = geometry_name.lower()

    if geometry_name == "holes":
        if mode == "force":
            dirichlet_bcs = [
                DirichletBC("bottom_x", pred_dict["bottom"], dof=0, val=0.0),
                DirichletBC("bottom_y", pred_dict["bottom"], dof=1, val=0.0)
            ]
            neumann_bcs = [
                NeumannBC("right", pred_dict["right"], direction=(1.0, 0.0), load_index=0),
                NeumannBC("top", pred_dict["top"], direction=(0.0, -1.0), load_index=1)
            ]
        else: # displacement
            dirichlet_bcs = [
                DirichletBC("bottom_x", pred_dict["bottom"], dof=0, val=0.0),
                DirichletBC("bottom_y", pred_dict["bottom"], dof=1, val=0.0),
                DirichletBC("top_y", pred_dict["top"], dof=1, val=0.0, is_prescribed=True)
            ]
            neumann_bcs = []
    else: # default "block"
        if mode == "force":
            dirichlet_bcs = [
                DirichletBC("left_x", pred_dict["left"], dof=0, val=0.0),
                DirichletBC("bottom_y", pred_dict["bottom"], dof=1, val=0.0)
            ]
            neumann_bcs = [
                NeumannBC("right", pred_dict["right"], direction=(-1.0, 0.0), load_index=0),
                NeumannBC("top", pred_dict["top"], direction=(0.0, -1.0), load_index=1)
            ]
        else: # displacement
            dirichlet_bcs = [
                DirichletBC("left_x", pred_dict["left"], dof=0, val=0.0),
                DirichletBC("bottom_y", pred_dict["bottom"], dof=1, val=0.0),
                DirichletBC("top_y", pred_dict["top"], dof=1, val=0.0, is_prescribed=True)
            ]
            neumann_bcs = []

    return BoundaryConditionConfig(mode=mode, dirichlet_bcs=dirichlet_bcs, neumann_bcs=neumann_bcs)


# ==============================================================================
# 3. Universal JAX-FEM HyperElasticity Problem & Adaptive Solver
# ==============================================================================

class HyperElasticityProblem(Problem):
    """
    Universal JAX-FEM HyperElasticity problem taking arbitrary stress function P(F)
    or parameterized internal variables for uncertainty propagation.
    """
    def __init__(
        self,
        piola_func: Callable,
        surface_maps: Optional[List[Callable]] = None,
        num_internal_params: int = 0,
        **kwargs
    ):
        self.piola_func = piola_func
        self._surface_maps = surface_maps or []
        self.num_internal_params = num_internal_params
        super().__init__(**kwargs)

    def custom_init(self):
        self.fe = self.fes[0]
        if self.num_internal_params > 0:
            self.internal_vars = [jnp.zeros((self.num_cells, self.fes[0].num_quads, self.num_internal_params), dtype=jnp.float64)]

    def set_params(self, params: jnp.ndarray):
        """Allows dynamic parameter updating without class re-instantiation or recompilation."""
        self.internal_vars = [jnp.tile(params[None, None, :], (self.num_cells, self.fes[0].num_quads, 1))]

    def get_surface_maps(self):
        return self._surface_maps

    def get_tensor_map(self):
        piola_fn = self.piola_func
        def first_PK_stress(u_grad, *args):
            I = jnp.eye(self.dim)
            F = u_grad + I
            if len(args) > 0 and args[0] is not None:
                return piola_fn(F, args[0])
            return piola_fn(F)
        return first_PK_stress


def solve_adaptive_fem(
    problem: Problem,
    bc_config: BoundaryConditionConfig,
    schedule: jnp.ndarray,
    petsc_options: dict,
    max_substeps: int = 32
) -> jnp.ndarray:
    """
    Solves nonlinear hyperelastic FEM over an incremental schedule (loads or displacements)
    with automated step-halving upon PETSc non-convergence.
    
    Parameters:
        problem: instantiated HyperElasticityProblem.
        bc_config: boundary condition configuration.
        schedule: (num_steps, n_dims) array of loads or displacements.
        petsc_options: solver dictionary.
        max_substeps: maximum substep divisions.
    Returns:
        u_array: (num_steps, num_nodes, 2)
    """
    u_list = []
    u = jnp.zeros_like(problem.mesh[0].points)
    n_steps = schedule.shape[0]
    dim = schedule.shape[1] if schedule.ndim > 1 else 1
    current_target = jnp.zeros(dim)

    # Cache prescribed Dirichlet indices in fe.vals_list
    prescribed_dof_map = []
    if bc_config.mode == "displacement":
        for i, dbc in enumerate(bc_config.dirichlet_bcs):
            if dbc.is_prescribed:
                prescribed_dof_map.append((i, dbc.dof))

    for step_idx in range(n_steps):
        target = schedule[step_idx]
        success = False
        current_u = u
        num_substeps = 1

        while not success:
            try:
                temp_u = current_u
                for sub_i in range(1, num_substeps + 1):
                    frac = sub_i / num_substeps
                    interm = current_target + frac * (target - current_target)

                    if bc_config.mode == "force":
                        # Apply Neumann surface tractions
                        surface_vars = []
                        for k, nbc in enumerate(bc_config.neumann_bcs):
                            num_b_cells = len(problem.boundary_inds_list[k])
                            shape_k = (num_b_cells, problem.fes[0].num_face_quads, 1)
                            val_k = interm[nbc.load_index]
                            surface_vars.append([jnp.full(fill_value=val_k, shape=shape_k)])
                        problem.internal_vars_surfaces = surface_vars
                    elif bc_config.mode == "displacement":
                        # Apply prescribed Dirichlet displacement
                        for val_idx, dof in prescribed_dof_map:
                            disp_val = interm[dof] if hasattr(interm, '__getitem__') else interm
                            problem.fes[0].vals_list[val_idx] = jnp.full_like(
                                problem.fes[0].vals_list[val_idx], disp_val
                            )

                    u_sol = solver(problem, solver_options={
                        'petsc_solver': petsc_options,
                        'initial_guess': temp_u
                    })
                    temp_u = u_sol[0]

                u = temp_u
                success = True
            except Exception as e:
                num_substeps *= 2
                if num_substeps > max_substeps:
                    raise RuntimeError(
                        f"FEM failed to converge at step {step_idx} (target={target}) "
                        f"with {max_substeps} sub-steps: {e}"
                    )

        current_target = target
        u_list.append(u)

    return jnp.stack(u_list, axis=0)


# ==============================================================================
# 4. Invariant Computation Helper
# ==============================================================================

def compute_all_invariants(
    F_array: jnp.ndarray,
    a0: Optional[np.ndarray] = None,
    a1: Optional[np.ndarray] = None
) -> Dict[str, np.ndarray]:
    """
    Computes isochoric invariants I1_bar, I2_bar, J, and anisotropic invariants
    I4_bar, I6_bar, I8_bar when fiber directions (a0, a1) are provided.
    F_array shape: (..., 2, 2) or (..., 3, 3)
    """
    orig_shape = F_array.shape
    if F_array.shape[-2:] == (2, 2):
        F_flat = F_array.reshape(-1, 2, 2)
        F_3d = np.zeros((F_flat.shape[0], 3, 3), dtype=np.float64)
        F_3d[:, :2, :2] = np.array(F_flat)
        F_3d[:, 2, 2] = 1.0
    else:
        F_3d = np.array(F_array).reshape(-1, 3, 3)

    I1_bar, I2_bar, J = compute_invariants_np(F_3d)

    out_shape = orig_shape[:-2]
    inv_dict = {
        "I1_bar": I1_bar.reshape(out_shape),
        "I2_bar": I2_bar.reshape(out_shape),
        "J": J.reshape(out_shape),
    }

    if a0 is not None:
        a0_arr = np.asarray(a0, dtype=np.float64)
        C = np.einsum('...ji,...jk->...ik', F_3d, F_3d)
        J_safe = np.clip(J, 1e-8, 1e8)
        C_bar = C / (J_safe**(2/3))[..., None, None]
        I4_bar = np.einsum('i,...ij,j->...', a0_arr, C_bar, a0_arr)
        inv_dict["I4_bar"] = I4_bar.reshape(out_shape)

        if a1 is not None:
            a1_arr = np.asarray(a1, dtype=np.float64)
            I6_bar = np.einsum('i,...ij,j->...', a1_arr, C_bar, a1_arr)
            inv_dict["I6_bar"] = I6_bar.reshape(out_shape)

            dot_a0_a1 = np.dot(a0_arr, a1_arr)
            I8_bar = dot_a0_a1 * np.einsum('i,...ij,j->...', a0_arr, C_bar, a1_arr)
            inv_dict["I8_bar"] = I8_bar.reshape(out_shape)

    return inv_dict


# ==============================================================================
# 5. Universal Dataset Exporter
# ==============================================================================

def export_fem_dataset(
    output_npz_path: str,
    mesh_pos: np.ndarray,
    cells: np.ndarray,
    node_type: np.ndarray,
    u_true: np.ndarray,
    loads_noisy: np.ndarray,
    disp_noise: float,
    load_noise_std: np.ndarray,
    load_noise_std_steps: np.ndarray,
    seed: int = 42,
    a0: Optional[np.ndarray] = None,
    a1: Optional[np.ndarray] = None,
    a2: Optional[np.ndarray] = None,
    mode: str = "force"
) -> Dict[str, Any]:
    """
    Computes kinematics and exports ground truth and observed fields into a clean .npz archive:
      1. Displacements: u_true (unperturbed) and u_obs / u (with noise on free DOFs).
      2. Kinematics: F_true, F_obs (and F), dNdX, dA, f_neu.
      3. Invariants:
         - true_I1_bar, true_I2_bar, true_J
         - obs_I1_bar, obs_I2_bar, obs_J
         - (true_I4_bar, true_I6_bar, true_I8_bar, obs_I4_bar, obs_I6_bar, obs_I8_bar if anisotropic).
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_npz_path)), exist_ok=True)
    num_steps = u_true.shape[0]
    rng = jax.random.PRNGKey(seed)

    u_obs_list = []
    F_true_list = []
    F_obs_list = []
    f_neu_list = []

    m_cells = mesh_pos[cells]

    for step in range(num_steps):
        rng, subkey_disp = jax.random.split(rng)
        u_step_true = u_true[step]

        # Add displacement noise to free nodes (where is_fix_x != 1 or is_fix_y != 1)
        u_noise = jax.random.normal(subkey_disp, u_step_true.shape) * disp_noise
        free_nodes = (node_type[:, 1] != 1) & (node_type[:, 2] != 1)
        u_noise = u_noise.at[~free_nodes].set(0.0)
        u_step_obs = u_step_true + u_noise

        u_cells_true = u_step_true[cells]
        u_cells_obs = u_step_obs[cells]
        node_type_cells = node_type[cells]

        F_step_true, dNdX = deformation_gradient_element(m_cells, u_cells_true)
        F_step_obs, _ = deformation_gradient_element(m_cells, u_cells_obs)
        dA = jnp.linalg.det(transformation_jacobian(m_cells)) / 2.0

        if mode == "force":
            load_step = loads_noisy[step]
            t3 = float(load_step[0]) if len(load_step) > 0 else 0.0
            t4 = float(load_step[1]) if len(load_step) > 1 else 0.0
            f_neu_cells = jax.vmap(neumann_cell_force, in_axes=(0, 0, None, None))(
                m_cells, node_type_cells, t3, t4
            )
            f_neu_step = jnp.zeros((mesh_pos.shape[0], 2), dtype=jnp.float64).at[cells].add(f_neu_cells)
        else:
            f_neu_step = jnp.zeros((mesh_pos.shape[0], 2), dtype=jnp.float64)

        u_obs_list.append(u_step_obs)
        F_true_list.append(F_step_true)
        F_obs_list.append(F_step_obs)
        f_neu_list.append(f_neu_step)

    u_obs_arr = jnp.stack(u_obs_list)
    F_true_arr = jnp.stack(F_true_list)
    F_obs_arr = jnp.stack(F_obs_list)
    f_neu_arr = jnp.stack(f_neu_list)

    # Compute invariants
    inv_true = compute_all_invariants(F_true_arr, a0=a0, a1=a1)
    inv_obs = compute_all_invariants(F_obs_arr, a0=a0, a1=a1)

    export_dict = {
        # Geometry & Mesh
        "mesh_pos": mesh_pos,
        "cells": cells,
        "node_type": node_type,
        "dNdX": dNdX,
        "dA": dA,
        "load": loads_noisy,
        "f_neu": f_neu_arr,
        "load_noise_std": load_noise_std,
        "load_noise_std_steps": load_noise_std_steps,

        # Displacements
        "u": u_obs_arr,       # Backward compatible key
        "u_obs": u_obs_arr,
        "u_true": u_true,

        # Kinematics
        "F": F_obs_arr,       # Backward compatible key
        "F_obs": F_obs_arr,
        "F_true": F_true_arr,

        # Invariants (True)
        "true_I1_bar": inv_true["I1_bar"],
        "true_I2_bar": inv_true["I2_bar"],
        "true_J": inv_true["J"],

        # Invariants (Observed)
        "obs_I1_bar": inv_obs["I1_bar"],
        "obs_I2_bar": inv_obs["I2_bar"],
        "obs_J": inv_obs["J"],
    }

    # Anisotropic invariants (only included when fiber vectors are defined)
    if a0 is not None:
        export_dict["a0"] = a0
        if "I4_bar" in inv_true:
            export_dict["true_I4_bar"] = inv_true["I4_bar"]
            export_dict["obs_I4_bar"] = inv_obs["I4_bar"]
    if a1 is not None:
        export_dict["a1"] = a1
        if "I6_bar" in inv_true:
            export_dict["true_I6_bar"] = inv_true["I6_bar"]
            export_dict["obs_I6_bar"] = inv_obs["I6_bar"]
        if "I8_bar" in inv_true:
            export_dict["true_I8_bar"] = inv_true["I8_bar"]
            export_dict["obs_I8_bar"] = inv_obs["I8_bar"]
    if a2 is not None:
        export_dict["a2"] = a2

    np.savez_compressed(output_npz_path, **export_dict)
    return export_dict
