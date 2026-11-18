"""
Finite-element bookkeeping shared by the GP extraction (HSGP, exact GP), the data generation and the FEM engine:
DOF index sets of a mesh, Neumann edge forces of a triangle, and the strong-ellipticity check of an energy.
"""
from typing import Any

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


def dof_sets(node_type) -> dict:
    """DOF index sets of a mesh (flat DOF = 2 * node + direction), from node_type columns
    [0: internal, 1: fixed x, 2: fixed y, 3: loaded x, 4: loaded y]:
      free:   DOFs that are neither fixed nor loaded (their internal force must vanish),
      rx, ry: loaded-edge DOFs, whose internal forces sum to the measured x / y reaction,
      fx, fy: fixed DOFs (the supports opposite the loaded edges)."""
    nt = np.asarray(node_type)
    is_fix_x, is_fix_y = nt[:, 1] == 1, nt[:, 2] == 1
    is_loaded_x, is_loaded_y = nt[:, 3] == 1, nt[:, 4] == 1
    free_mask = np.stack([~(is_fix_x | is_loaded_x), ~(is_fix_y | is_loaded_y)], axis=-1).reshape(-1)
    return dict(free=np.where(free_mask)[0], rx=2 * np.where(is_loaded_x)[0], ry=2 * np.where(is_loaded_y)[0] + 1,
                fx=2 * np.where(is_fix_x)[0], fy=2 * np.where(is_fix_y)[0] + 1)


def neumann_cell_force(coords_el: jnp.ndarray, onehot_types_el: jnp.ndarray, t3: float, t4: float):
    """
    Nodal forces (3, 2) of a uniform edge traction on one triangle.
    onehot_types_el: (3, 5) one-hot node types [0: internal, 1: fixed x, 2: fixed y, 3: right (t3), 4: top (t4)].
    """
    edges = jnp.array([[0, 1], [1, 2], [2, 0]])
    f_cell = jnp.zeros((3, 2), dtype=jnp.float64)

    for idx in range(3):
        i, j = edges[idx]
        is_right = (onehot_types_el[i, 3] == 1) & (onehot_types_el[j, 3] == 1)
        is_top = (onehot_types_el[i, 4] == 1) & (onehot_types_el[j, 4] == 1)
        L = jnp.linalg.norm(coords_el[j] - coords_el[i])

        f_cell = f_cell.at[i, 0].add(jnp.where(is_right, 0.5 * L * t3, 0.0))
        f_cell = f_cell.at[j, 0].add(jnp.where(is_right, 0.5 * L * t3, 0.0))
        f_cell = f_cell.at[i, 1].add(jnp.where(is_top, 0.5 * L * t4, 0.0))
        f_cell = f_cell.at[j, 1].add(jnp.where(is_top, 0.5 * L * t4, 0.0))

    return f_cell


def ellipticity_penalty(psi_fn: Any, f3x3: jnp.ndarray, n_dirs: int = 8):
    """
    Strong-ellipticity (material stability) check of an energy psi at the given deformation states.
    For each state and direction n the in-plane acoustic tensor Q_ik(n) = A_ijkl n_j n_l (A = dP/dF, out-of-plane
    F entries held fixed) must be positive definite. Its smallest eigenvalue is normalised by the mean stiffness
    (so the result does not depend on the stress scale) and only negative values are penalised:
        penalty = sum_{states, n} relu(-lambda_min(Q) / s_ref)^2,   zero for any stable material.
    f3x3: (..., 3, 3). Returns (penalty, fraction of (state, direction) pairs that violate ellipticity).
    """
    def tangent(f3):
        return jax.jacfwd(jax.grad(lambda f2: psi_fn(f3.at[:2, :2].set(f2))))(f3[:2, :2])   # (2,2,2,2)

    # batched: a vmap over all states holds every state's second-derivative intermediates at once
    A = jax.lax.map(tangent, f3x3.reshape(-1, 3, 3), batch_size=256)
    theta = jnp.arange(n_dirs) * jnp.pi / n_dirs
    n = jnp.stack([jnp.cos(theta), jnp.sin(theta)], axis=-1)                          # (D, 2)
    Q = jnp.einsum("eijkl,dj,dl->edik", A, n, n)
    a, c = Q[..., 0, 0], Q[..., 1, 1]
    b = 0.5 * (Q[..., 0, 1] + Q[..., 1, 0])
    lam_min = 0.5 * (a + c) - jnp.sqrt(0.25 * (a - c) ** 2 + b ** 2 + 1e-30)
    s_ref = jax.lax.stop_gradient(jnp.abs(jnp.mean(0.5 * (a + c))) + 1e-12)
    r = lam_min / s_ref
    return jnp.sum(jax.nn.relu(-r) ** 2), jnp.mean(r < 0)
