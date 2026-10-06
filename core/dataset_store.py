"""
Clean-dataset store: one seed-independent FEM solution per physical configuration, with measurement
noise added deterministically at load time.

A synthetic experiment is fully described by
  * a clean dataset  (mesh, true displacements, true loads/reactions) -- solved and stored once, and
  * an observation   (seed, disp_noise, load_noise)                   -- regenerated on demand.

Both are carried by one string, the dataset spec, which can be passed wherever a dataset path was used:

    dataset/clean/isihara_block_disp_pstrain_3f9c1a2e.npz?seed=3&disp_noise=0.0005&load_noise=0.05

`load_dataset(spec)` returns the same keys the former per-seed files stored, and seed k reproduces the
former seed-k file (same PRNG sequence). Legacy full per-seed files load unchanged.
"""
import hashlib
import json
import os
from typing import Any, Callable, Dict, Optional, Tuple
from urllib.parse import parse_qs, urlencode

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from core.utils import compute_invariants_np, deformation_gradient_element, transformation_jacobian
from core.loss_function import neumann_cell_force

CLEAN_FORMAT = "clean_v1"
CLEAN_DIR = "dataset/clean"


# ------------------------------------------------------------------------------------------------
# Naming
# ------------------------------------------------------------------------------------------------
def config_hash(config: Dict[str, Any]) -> str:
    """Short stable hash of the full physical configuration (key order and float formatting independent)."""
    canonical = json.dumps(config, sort_keys=True, default=lambda x: np.asarray(x).tolist())
    return hashlib.sha1(canonical.encode()).hexdigest()[:8]


def clean_dataset_path(config: Dict[str, Any], root: str = CLEAN_DIR) -> str:
    """Readable prefix (model, geometry, modes) + hash of everything, so distinct configurations never collide."""
    control = {"displacement": "disp", "force": "force"}[config["control_mode"]]
    stress = {"plane_strain": "pstrain", "plane_stress": "pstress"}[config["stress_mode"]]
    name = f"{config['material_model']}_{config['geometry']}_{control}_{stress}_{config_hash(config)}.npz"
    return os.path.join(root, name)


def make_dataset_spec(clean_path: str, seed: int, disp_noise: float, load_noise: float) -> str:
    return f"{clean_path}?{urlencode({'seed': int(seed), 'disp_noise': float(disp_noise), 'load_noise': float(load_noise)})}"


def parse_dataset_spec(spec: str) -> Tuple[str, Dict[str, float]]:
    """Splits 'path?seed=..&disp_noise=..&load_noise=..' into (path, observation params); plain paths give {}."""
    path, _, query = str(spec).partition("?")
    params = {k: v[0] for k, v in parse_qs(query).items()}
    out = {}
    if "seed" in params:
        out["seed"] = int(params["seed"])
    for k in ("disp_noise", "load_noise"):
        if k in params:
            out[k] = float(params[k])
    return path, out


def dataset_exists(spec: Optional[str]) -> bool:
    return bool(spec) and os.path.exists(parse_dataset_spec(spec)[0])


# ------------------------------------------------------------------------------------------------
# Kinematics
# ------------------------------------------------------------------------------------------------
def compute_all_invariants(
    F_array: jnp.ndarray,
    a0: Optional[np.ndarray] = None,
    a1: Optional[np.ndarray] = None,
    lam3: Optional[np.ndarray] = None
) -> Dict[str, np.ndarray]:
    """
    Computes isochoric invariants I1_bar, I2_bar, J, and anisotropic invariants
    I4_bar, I6_bar, I8_bar when fiber directions (a0, a1) are provided.
    F_array shape: (..., 2, 2) or (..., 3, 3)
    lam3: optional out-of-plane stretch array of shape (...) when F_array is (..., 2, 2).
    """
    orig_shape = F_array.shape
    if F_array.shape[-2:] == (2, 2):
        F_flat = F_array.reshape(-1, 2, 2)
        F_3d = np.zeros((F_flat.shape[0], 3, 3), dtype=np.float64)
        F_3d[:, :2, :2] = np.array(F_flat)
        if lam3 is not None:
            lam3_flat = np.asarray(lam3, dtype=np.float64).reshape(-1)
            F_3d[:, 2, 2] = lam3_flat
        else:
            F_3d[:, 2, 2] = 1.0
    else:
        F_3d = np.array(F_array).reshape(-1, 3, 3)

    I1_bar, I2_bar, J = compute_invariants_np(F_3d)

    out_shape = orig_shape[:-2]
    inv_dict = {
        "I1_bar": I1_bar.reshape(out_shape),
        "I2_bar": I2_bar.reshape(out_shape),
        "J": J.reshape(out_shape),
        "F_3d": F_3d.reshape(*out_shape, 3, 3)
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


def displacement_reactions(u_true: np.ndarray, mesh_pos: np.ndarray, cells: np.ndarray, node_type: np.ndarray,
                           piola_func_2d: Callable) -> np.ndarray:
    """True x/y reaction forces on the prescribed-displacement boundaries: (T, 2)."""
    m_cells = mesh_pos[cells]
    dA = jnp.linalg.det(transformation_jacobian(m_cells)) / 2.0
    out = []
    for step in range(u_true.shape[0]):
        F_step, dNdX = deformation_gradient_element(m_cells, u_true[step][cells])
        P_step = jax.vmap(piola_func_2d)(F_step)
        f_int_cell = jnp.swapaxes(jnp.einsum("cij, cnj -> cin", P_step, dNdX) * dA[:, None, None], 1, 2)
        f_int_nodes = jnp.zeros((mesh_pos.shape[0], 2), dtype=jnp.float64).at[cells].add(f_int_cell)
        out.append([float(jnp.sum(f_int_nodes[node_type[:, 3] == 1, 0])),
                    float(jnp.sum(f_int_nodes[node_type[:, 4] == 1, 1]))])
    return np.array(out)


# ------------------------------------------------------------------------------------------------
# Clean datasets
# ------------------------------------------------------------------------------------------------
def save_clean_dataset(path: str, config: Dict[str, Any], mesh_pos: np.ndarray, cells: np.ndarray,
                       node_type: np.ndarray, u_true: np.ndarray, loads_true: np.ndarray,
                       reaction_forces_true: Optional[np.ndarray] = None, lam3_true: Optional[np.ndarray] = None,
                       a0=None, a1=None, a2=None) -> str:
    """
    Stores the seed-independent truth of one configuration.
    loads_true: (T, 2) applied tractions (force control) or prescribed displacements (displacement control).
    lam3_true: (T, C) out-of-plane stretch of the TRUE state (plane stress only).
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    arrays = dict(format=CLEAN_FORMAT, config_json=json.dumps(config, sort_keys=True, default=lambda x: np.asarray(x).tolist()),
                  mesh_pos=mesh_pos, cells=cells, node_type=node_type, u_true=np.asarray(u_true),
                  loads_true=np.asarray(loads_true), control_mode=config["control_mode"], stress_mode=config["stress_mode"])
    for k, v in dict(reaction_forces_true=reaction_forces_true, lam3_true=lam3_true, a0=a0, a1=a1, a2=a2).items():
        if v is not None:
            arrays[k] = np.asarray(v)
    np.savez_compressed(path, **arrays)
    return path


def observe_dataset(clean: Dict[str, Any], seed: int, disp_noise: float, load_noise: float,
                    solve_lambda3_fn: Optional[Callable] = None) -> Dict[str, Any]:
    """
    Adds measurement noise to a clean dataset and derives all kinematic fields.
    Returns the key set of the former per-seed files. The PRNG sequence matches the former generator,
    so a given seed reproduces the former seed-specific file.
    Plane stress additionally needs solve_lambda3_fn for the observed out-of-plane stretch.
    """
    mesh_pos, cells, node_type = clean["mesh_pos"], clean["cells"], clean["node_type"]
    u_true = np.asarray(clean["u_true"])
    mode, stress_mode = str(clean["control_mode"]), str(clean["stress_mode"])
    loads_true = np.asarray(clean["loads_true"])
    a0, a1, a2 = (clean[k] if k in clean else None for k in ("a0", "a1", "a2"))
    num_steps = u_true.shape[0]
    m_cells = mesh_pos[cells]

    if mode == "force":
        # Former generator: one draw scales the whole load ramp, key = PRNGKey(seed)
        z = float(jax.random.normal(jax.random.PRNGKey(seed)))
        loads_noisy = loads_true * (1.0 + load_noise * z)
        load_noise_std = load_noise * loads_true
        load_noise_std_steps = load_noise_std * np.linspace(0, 1, num_steps).reshape(-1, 1)
    else:
        loads_noisy = np.zeros((num_steps, 2))

    if stress_mode == "plane_stress" and solve_lambda3_fn is None:
        raise ValueError("Plane-stress observations need solve_lambda3_fn (the observed lambda3 depends on the material).")

    rng = jax.random.PRNGKey(seed)
    free_nodes = (node_type[:, 1] != 1) & (node_type[:, 2] != 1)
    if mode == "displacement":
        free_nodes = free_nodes & (node_type[:, 3] != 1) & (node_type[:, 4] != 1)
    dA = jnp.linalg.det(transformation_jacobian(m_cells)) / 2.0

    u_obs_list, F_true_list, F_obs_list, f_neu_list, lam3_obs_list = [], [], [], [], []
    for step in range(num_steps):
        rng, subkey_disp = jax.random.split(rng)
        u_noise = (jax.random.normal(subkey_disp, u_true[step].shape) * disp_noise).at[~free_nodes].set(0.0)
        u_step_obs = u_true[step] + u_noise

        F_step_true, dNdX = deformation_gradient_element(m_cells, u_true[step][cells])
        F_step_obs, _ = deformation_gradient_element(m_cells, u_step_obs[cells])

        if mode == "force":
            f_neu_cells = jax.vmap(neumann_cell_force, in_axes=(0, 0, None, None))(
                m_cells, node_type[cells], float(loads_noisy[step][0]), float(loads_noisy[step][1]))
            f_neu_step = jnp.zeros((mesh_pos.shape[0], 2), dtype=jnp.float64).at[cells].add(f_neu_cells)
        else:
            f_neu_step = jnp.zeros((mesh_pos.shape[0], 2), dtype=jnp.float64)

        if stress_mode == "plane_stress":
            lam3_obs_list.append(np.array(jax.vmap(solve_lambda3_fn)(F_step_obs)))

        u_obs_list.append(u_step_obs)
        F_true_list.append(F_step_true)
        F_obs_list.append(F_step_obs)
        f_neu_list.append(f_neu_step)

    u_obs_arr = jnp.stack(u_obs_list)
    F_true_arr, F_obs_arr = jnp.stack(F_true_list), jnp.stack(F_obs_list)
    lam3_true_arr = np.asarray(clean["lam3_true"]) if "lam3_true" in clean else None
    lam3_obs_arr = np.array(lam3_obs_list) if lam3_obs_list else None
    inv_true = compute_all_invariants(F_true_arr, a0=a0, a1=a1, lam3=lam3_true_arr)
    inv_obs = compute_all_invariants(F_obs_arr, a0=a0, a1=a1, lam3=lam3_obs_arr)

    if mode == "displacement":
        reactions_true = np.asarray(clean["reaction_forces_true"])
        rng, subkey_r1 = jax.random.split(rng)
        rng, subkey_r2 = jax.random.split(rng)
        noise_x = load_noise * np.abs(reactions_true[:, 0]) * np.array(jax.random.normal(subkey_r1, (num_steps,)))
        noise_y = load_noise * np.abs(reactions_true[:, 1]) * np.array(jax.random.normal(subkey_r2, (num_steps,)))
        loads_noisy = np.column_stack([reactions_true[:, 0] + noise_x, reactions_true[:, 1] + noise_y])
        load_noise_std = load_noise * np.abs(reactions_true)
        load_noise_std_steps = load_noise_std

    out = {
        "mesh_pos": mesh_pos, "cells": cells, "node_type": node_type, "dNdX": dNdX, "dA": dA,
        "load": loads_noisy, "f_neu": jnp.stack(f_neu_list),
        "load_noise_std": load_noise_std, "load_noise_std_steps": load_noise_std_steps,
        "u": u_obs_arr, "u_obs": u_obs_arr, "u_true": u_true,
        "F": F_obs_arr, "F_obs": F_obs_arr, "F_true": F_true_arr,
        "F_3d": inv_obs["F_3d"], "F_3d_true": inv_true["F_3d"],
        "true_I1_bar": inv_true["I1_bar"], "true_I2_bar": inv_true["I2_bar"], "true_J": inv_true["J"],
        "obs_I1_bar": inv_obs["I1_bar"], "obs_I2_bar": inv_obs["I2_bar"], "obs_J": inv_obs["J"],
        "stress_mode": stress_mode, "control_mode": mode,
        "seed": seed, "disp_noise": disp_noise, "load_noise": load_noise,
    }
    if lam3_obs_arr is not None:
        out["lam3"], out["lam3_true"] = lam3_obs_arr, lam3_true_arr
    if mode == "displacement":
        out["reaction_forces"], out["reaction_forces_true"] = loads_noisy, reactions_true
    if a0 is not None:
        out["a0"] = a0
        if "I4_bar" in inv_true:
            out["true_I4_bar"], out["obs_I4_bar"] = inv_true["I4_bar"], inv_obs["I4_bar"]
    if a1 is not None:
        out["a1"] = a1
        if "I6_bar" in inv_true:
            out["true_I6_bar"], out["obs_I6_bar"] = inv_true["I6_bar"], inv_obs["I6_bar"]
        if "I8_bar" in inv_true:
            out["true_I8_bar"], out["obs_I8_bar"] = inv_true["I8_bar"], inv_obs["I8_bar"]
    if a2 is not None:
        out["a2"] = a2
    # Same array types as the former np.load'ed files (consumers index with lists, mutate in place, ...)
    return {k: np.asarray(v) if isinstance(v, jax.Array) else v for k, v in out.items()}


class DatasetDict(dict):
    """dict with the `.files` attribute of np.load results, so existing consumers keep working."""
    @property
    def files(self):
        return list(self.keys())


def _plane_stress_lambda3_solver(config: Dict[str, Any]) -> Callable:
    """Rebuilds lambda3(F_2D) of the generating material (synthetic plane stress only; this uses the truth)."""
    from core.material_models import get_material
    from core.fem_engine import make_plane_stress_piola
    _, solve_lambda3 = make_plane_stress_piola(get_material(config["material_model"], **config.get("material_kwargs", {})))
    return solve_lambda3


def load_dataset(spec: str, seed: Optional[int] = None, disp_noise: Optional[float] = None,
                 load_noise: Optional[float] = None) -> DatasetDict:
    """
    Loads a dataset spec. Clean datasets are observed with (seed, disp_noise, load_noise) from the spec,
    overridable by the keyword arguments; legacy per-seed files are returned as stored.
    """
    path, params = parse_dataset_spec(spec)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")
    raw = np.load(path, allow_pickle=True)
    if "format" not in raw.files or str(raw["format"]) != CLEAN_FORMAT:
        return DatasetDict({k: raw[k] for k in raw.files})

    clean = {k: raw[k] for k in raw.files}
    seed = seed if seed is not None else params.get("seed")
    disp_noise = disp_noise if disp_noise is not None else params.get("disp_noise")
    load_noise = load_noise if load_noise is not None else params.get("load_noise")
    if seed is None or disp_noise is None or load_noise is None:
        raise ValueError(f"Clean dataset {path} needs seed, disp_noise and load_noise "
                         f"(e.g. '{make_dataset_spec(path, 1, 1e-4, 0.01)}').")
    config = json.loads(str(clean["config_json"]))
    solve_l3 = _plane_stress_lambda3_solver(config) if str(clean["stress_mode"]) == "plane_stress" else None
    obs = observe_dataset(clean, seed, disp_noise, load_noise, solve_lambda3_fn=solve_l3)
    obs["config_json"] = clean["config_json"]
    obs["dataset_spec"] = make_dataset_spec(path, seed, disp_noise, load_noise)
    return DatasetDict(obs)
