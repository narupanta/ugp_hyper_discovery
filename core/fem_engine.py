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

import logging
# jax_fem logs every Newton iteration at INFO/DEBUG (~30 MB per FEM validation worker log);
# keep warnings only unless JAX_FEM_LOG_LEVEL asks for more.
logging.getLogger("jax_fem").setLevel(os.environ.get("JAX_FEM_LOG_LEVEL", "WARNING").upper())

from core.utils import (
    fto3x3,
    deformation_gradient_element,
    transformation_jacobian,
    compute_invariants_np,
)
from core.loss_function import neumann_cell_force
from core.dataset_store import compute_all_invariants  # noqa: F401  (re-export)
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


class TTCGeometry(BaseGeometry):
    """
    Tensile test specimen TTc with 3 circular holes (Abbasi et al. 2026, EUCLID experimental data).
    Physical bounds: X in [-90.33, -22.25] mm, Y in [-25.74, 23.89] mm.
    Left clamp at X <= -90.0 mm (bcx=1), Right clamp at X >= -22.9 mm (bcx=2).
    """
    def __init__(self, mesh_npz: str = "mesh/ttc_mesh.npz"):
        self.mesh_npz = mesh_npz

    def generate_mesh(self, msh_path: str, npz_path: str) -> None:
        if os.path.exists(self.mesh_npz):
            import shutil
            os.makedirs(os.path.dirname(os.path.abspath(npz_path)), exist_ok=True)
            shutil.copyfile(self.mesh_npz, npz_path)
        else:
            raise FileNotFoundError(f"TTC reference mesh not found at {self.mesh_npz}. Please run dataset/convert_ttc_to_pipeline_npz.py first.")

    def get_boundary_predicates(self) -> Dict[str, Callable[[jnp.ndarray], jnp.ndarray]]:
        return {
            "left": lambda pt: pt[0] <= -90.0,
            "right": lambda pt: pt[0] >= -22.9,
            "bottom": lambda pt: pt[1] <= -25.5,
            "top": lambda pt: pt[1] >= 23.5,
        }


_GEOMETRY_REGISTRY: Dict[str, Callable[..., BaseGeometry]] = {
    "block": BlockGeometry,
    "holes": HolesGeometry,
    "ttc": TTCGeometry,
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
    schedule_index: index in the displacement schedule vector.
    """
    def __init__(self, name: str, location_fn: Callable, dof: int, 
                 val: Union[float, Callable] = 0.0, is_prescribed: bool = False,
                 schedule_index: int = 0):
        self.name = name
        self.location_fn = location_fn
        self.dof = int(dof)
        self.val = val
        self.is_prescribed = is_prescribed
        self.schedule_index = int(schedule_index)


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
        [is_internal, is_fix_x, is_fix_y, is_loaded_x, is_loaded_y]
        Completely unified across force and displacement modes:
        - Channels 1, 2: zero-fixed Dirichlet DOFs (u_x=0, u_y=0)
        - Channels 3, 4: loaded boundaries (traction DOFs in force mode, prescribed DOFs in displacement mode)
        - Channel 0: internal nodes
        """
        n_nodes = node_coords.shape[0]
        is_fix_x = jnp.zeros(n_nodes, dtype=bool)
        is_fix_y = jnp.zeros(n_nodes, dtype=bool)
        is_loaded_x = jnp.zeros(n_nodes, dtype=bool)
        is_loaded_y = jnp.zeros(n_nodes, dtype=bool)

        # 1. Evaluate Dirichlet DOFs
        for dbc in self.dirichlet_bcs:
            mask = jax.vmap(dbc.location_fn)(node_coords)
            if dbc.is_prescribed:
                if dbc.dof == 0:
                    is_loaded_x = is_loaded_x | mask
                elif dbc.dof == 1:
                    is_loaded_y = is_loaded_y | mask
            else:
                if dbc.dof == 0:
                    is_fix_x = is_fix_x | mask
                elif dbc.dof == 1:
                    is_fix_y = is_fix_y | mask

        # 2. Evaluate Neumann Traction DOFs (force mode)
        for nbc in self.neumann_bcs:
            mask = jax.vmap(nbc.location_fn)(node_coords)
            if abs(nbc.direction[0]) > 1e-6:
                is_loaded_x = is_loaded_x | mask
            if abs(nbc.direction[1]) > 1e-6:
                is_loaded_y = is_loaded_y | mask

        is_internal = ~(is_fix_x | is_fix_y | is_loaded_x | is_loaded_y)
        node_type = jnp.stack([
            is_internal, is_fix_x, is_fix_y, is_loaded_x, is_loaded_y
        ], axis=-1).astype(jnp.float32)
        return node_type


def create_default_bc_config(
    geometry_name: str = "block",
    mode: str = "force",
    pred_dict: Optional[Dict[str, Callable]] = None,
    prescribe_right: Optional[bool] = None,
    clamp_top_x: Optional[bool] = None,
    node_coords: Optional[jnp.ndarray] = None,
    u_boundary: Optional[jnp.ndarray] = None
) -> BoundaryConditionConfig:
    """
    Factory helper creating standard boundary conditions for standard geometries.
    """
    if pred_dict is None:
        geom = get_geometry(geometry_name)
        pred_dict = geom.get_boundary_predicates()

    mode = mode.lower()
    geometry_name = geometry_name.lower()

    if prescribe_right is None:
        # Default: block is biaxial (prescribe_right=True), holes is uniaxial tension (prescribe_right=False)
        prescribe_right = (geometry_name != "holes")

    if clamp_top_x is None:
        # Default: clamp top x only for holes in displacement mode
        clamp_top_x = (geometry_name == "holes" and mode == "displacement")
    elif geometry_name != "holes":
        # Strictly enforce that clamp_top_x is never applied to block
        clamp_top_x = False

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
            ]
            if clamp_top_x:
                dirichlet_bcs.append(DirichletBC("top_x", pred_dict["top"], dof=0, val=0.0, is_prescribed=False))
            sched_idx = 0
            if prescribe_right:
                dirichlet_bcs.append(DirichletBC("right_x", pred_dict["right"], dof=0, val=0.0, is_prescribed=True, schedule_index=sched_idx))
                sched_idx += 1
            dirichlet_bcs.append(DirichletBC("top_y", pred_dict["top"], dof=1, val=0.0, is_prescribed=True, schedule_index=sched_idx))
            neumann_bcs = []
    elif geometry_name == "ttc":
        # TTc tensile specimen with 3 holes
        # Left boundary (X <= -90.0) is clamped grip (bcx == 1)
        # Right boundary (X >= -22.9) is loaded / pulled grip (bcx == 2)
        if node_coords is not None and u_boundary is not None:
            ref_pts = jnp.asarray(node_coords)
            u_pts = jnp.asarray(u_boundary)

            def make_val_fn(dof):
                def val_fn(point):
                    dists = jnp.sum((ref_pts - point)**2, axis=-1)
                    idx = jnp.argmin(dists)
                    return u_pts[idx, dof]
                return val_fn

            val_left_x = make_val_fn(0)
            val_left_y = make_val_fn(1)
            val_right_x = make_val_fn(0)
            val_right_y = make_val_fn(1)
        else:
            val_left_x = 0.0
            val_left_y = 0.0
            val_right_x = 0.0
            val_right_y = 0.0

        if mode == "force":
            dirichlet_bcs = [
                DirichletBC("left_x", pred_dict["left"], dof=0, val=val_left_x),
                DirichletBC("left_y", pred_dict["left"], dof=1, val=val_left_y)
            ]
            neumann_bcs = [
                NeumannBC("right", pred_dict["right"], direction=(-1.0, 0.0), load_index=0)
            ]
        else: # displacement
            dirichlet_bcs = [
                DirichletBC("left_x", pred_dict["left"], dof=0, val=val_left_x, is_prescribed=False),
                DirichletBC("left_y", pred_dict["left"], dof=1, val=val_left_y, is_prescribed=False),
                DirichletBC("right_x", pred_dict["right"], dof=0, val=val_right_x, is_prescribed=True, schedule_index=0),
                DirichletBC("right_y", pred_dict["right"], dof=1, val=val_right_y, is_prescribed=False)
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
            ]
            sched_idx = 0
            if prescribe_right:
                dirichlet_bcs.append(DirichletBC("right_x", pred_dict["right"], dof=0, val=0.0, is_prescribed=True, schedule_index=sched_idx))
                sched_idx += 1
            dirichlet_bcs.append(DirichletBC("top_y", pred_dict["top"], dof=1, val=0.0, is_prescribed=True, schedule_index=sched_idx))
            neumann_bcs = []

    return BoundaryConditionConfig(mode=mode, dirichlet_bcs=dirichlet_bcs, neumann_bcs=neumann_bcs)


def make_plane_stress_piola(mat_model: Any, max_iter: int = 15) -> Tuple[Callable, Callable]:
    """
    Constructs an autodiff-compatible 2D First Piola-Kirchhoff stress function
    P_2D(F_2D) and lambda_3 root solver satisfying P_33(F_2D, lambda_3) = 0
    for compressible hyperelastic materials.
    """
    def solve_lambda3(F_2d: jnp.ndarray) -> jnp.ndarray:
        # Starting at 1.0 avoids overshooting into clipping bounds during severe compression (e.g. Equibiaxial Compression)
        lam3_0 = 1.0

        def step_fn(i, lam):
            def p33_val(l):
                F_3d = jnp.array([
                    [F_2d[0, 0], F_2d[0, 1], 0.0],
                    [F_2d[1, 0], F_2d[1, 1], 0.0],
                    [0.0,        0.0,        l]
                ])
                return mat_model.P(F_3d)[2, 2]

            p33, dp33 = jax.value_and_grad(p33_val)(lam)
            lam_next = lam - p33 / jnp.where(jnp.abs(dp33) < 1e-12, 1.0, dp33)
            return jnp.clip(lam_next, 1e-3, 100.0)

        return jax.lax.fori_loop(0, max_iter, step_fn, lam3_0)

    def piola_2d(F_2d: jnp.ndarray) -> jnp.ndarray:
        lam3 = solve_lambda3(F_2d)
        F_3d = jnp.array([
            [F_2d[0, 0], F_2d[0, 1], 0.0],
            [F_2d[1, 0], F_2d[1, 1], 0.0],
            [0.0,        0.0,        lam3]
        ])
        return mat_model.P(F_3d)[:2, :2]

    return piola_2d, solve_lambda3


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
        max_newton_iters: int = 50,
        **kwargs
    ):
        self.piola_func = piola_func
        self._surface_maps = surface_maps or []
        self.num_internal_params = num_internal_params
        self.max_newton_iters = max_newton_iters
        self._newton_iter_count = 0
        super().__init__(**kwargs)

    def custom_init(self):
        self.fe = self.fes[0]
        if self.num_internal_params > 0:
            self.internal_vars = [jnp.zeros((self.num_cells, self.fes[0].num_quads, self.num_internal_params), dtype=jnp.float64)]

    def set_params(self, params: jnp.ndarray):
        """Allows dynamic parameter updating without class re-instantiation or recompilation."""
        self.internal_vars = [jnp.tile(params[None, None, :], (self.num_cells, self.fes[0].num_quads, 1))]

    def reset_newton_counter(self):
        self._newton_iter_count = 0

    def newton_update(self, sol_list):
        self._newton_iter_count += 1
        if self._newton_iter_count > self.max_newton_iters:
            raise RuntimeError(
                f"Newton solver exceeded maximum allowed iterations ({self.max_newton_iters}) without converging."
            )
        return super().newton_update(sol_list)

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
    petsc_options: Optional[Dict] = None,
    max_substeps: int = 1,
    initial_substeps: int = 1,
    u_boundary_steps: Optional[jnp.ndarray] = None
) -> jnp.ndarray:
    """
    Solves nonlinear hyperelastic FEM over an incremental schedule (loads or displacements).
    
    Parameters:
        problem: instantiated HyperElasticityProblem.
        bc_config: boundary condition configuration.
        schedule: (num_steps, n_dims) array of loads or displacements.
        petsc_options: solver dictionary.
        max_substeps: maximum substep divisions per loadstep (default: 1, no step-halving).
        initial_substeps: initial number of substeps per schedule increment (default: 1, exactly 1 solve per step).
        u_boundary_steps: optional (num_steps, num_nodes, 2) array of full boundary displacement fields.
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
                prescribed_dof_map.append((i, dbc.schedule_index))

    base_vals_list = [jnp.array(v) for v in problem.fes[0].vals_list]

    for step_idx in range(n_steps):
        target = schedule[step_idx]
        success = False
        current_u = u
        num_substeps = initial_substeps

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
                        if u_boundary_steps is not None and u_boundary_steps.ndim == 3 and len(u_boundary_steps) == n_steps:
                            # Direct multi-step experimental boundary ramping from step k-1 to step k
                            fe = problem.fes[0]
                            for i_dbc, dbc in enumerate(bc_config.dirichlet_bcs):
                                n_inds = fe.node_inds_list[i_dbc]
                                dof = fe.vec_inds_list[i_dbc][0]
                                curr_b = u_boundary_steps[step_idx, n_inds, dof]
                                prev_b = u_boundary_steps[step_idx - 1, n_inds, dof] if step_idx > 0 else jnp.zeros_like(curr_b)
                                fe.vals_list[i_dbc] = prev_b + frac * (curr_b - prev_b)
                        else:
                            # Apply prescribed Dirichlet displacement
                            for val_idx, s_idx in prescribed_dof_map:
                                disp_val = interm[s_idx] if (hasattr(interm, '__getitem__') and interm.ndim > 0) else interm
                                base_v = base_vals_list[val_idx]
                                if jnp.all(base_v == 0.0):
                                    problem.fes[0].vals_list[val_idx] = jnp.full_like(base_v, disp_val)
                                else:
                                    problem.fes[0].vals_list[val_idx] = base_v * frac

                            # Scale non-zero non-prescribed Dirichlet boundaries (e.g. experimental clamps)
                            for i_dbc, dbc in enumerate(bc_config.dirichlet_bcs):
                                if not dbc.is_prescribed:
                                    base_v = base_vals_list[i_dbc]
                                    if not jnp.all(base_v == 0.0):
                                        problem.fes[0].vals_list[i_dbc] = base_v * frac

                    if hasattr(problem, 'reset_newton_counter'):
                        problem.reset_newton_counter()

                    u_sol = solver(problem, solver_options={
                        'petsc_solver': petsc_options,
                        'initial_guess': temp_u
                    })
                    temp_u = u_sol[0]

                u = temp_u
                success = True
            except Exception as e:
                if num_substeps >= max_substeps:
                    raise RuntimeError(
                        f"FEM failed to converge at step {step_idx + 1}/{n_steps} (target={target}) "
                        f"without sub-stepping: {e}"
                    )
                num_substeps *= 2
                if num_substeps > max_substeps:
                    raise RuntimeError(
                        f"FEM failed to converge at step {step_idx + 1}/{n_steps} (target={target}) "
                        f"with {max_substeps} sub-steps: {e}"
                    )

        current_target = target
        u_list.append(u)

    return jnp.stack(u_list, axis=0)


# ==============================================================================
# 4. Invariant Computation Helper
# ==============================================================================

# compute_all_invariants now lives in core.dataset_store (re-exported here for existing imports).
# Dataset export is split into clean datasets + observation: see core.dataset_store.
