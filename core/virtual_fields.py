import jax
import jax.numpy as jnp
import numpy as np
from typing import Tuple, Optional

# Enforce mandatory 64-bit precision standard
jax.config.update("jax_enable_x64", True)


def build_kinematic_virtual_fields(
    mesh_pos: np.ndarray | jnp.ndarray,
    node_type: np.ndarray | jnp.ndarray,
    order: int = 2
) -> jnp.ndarray:
    """
    Constructs an orthonormal basis of kinematically admissible virtual displacement fields
    vanishing on Dirichlet boundaries.
    
    Args:
        mesh_pos: (N_nodes, 2) nodal coordinates.
        node_type: (N_nodes, >=3) boundary flags:
                   column 1: is_fix_x (Dirichlet constraint in x)
                   column 2: is_fix_y (Dirichlet constraint in y)
        order: Polynomial degree of virtual field basis (default 2 -> 12 fields).
        
    Returns:
        V_basis: (M, N_nodes, 2) jnp.ndarray with dtype float64, satisfying:
                 1. Vanishing on Dirichlet boundary: V_basis[:, is_fix_x, 0] == 0, V_basis[:, is_fix_y, 1] == 0
                 2. Orthonormality on free DOFs: sum_{i, d} V_basis[m, i, d] * V_basis[k, i, d] == delta_mk
    """
    pos_np = np.asarray(mesh_pos, dtype=np.float64)
    node_type_np = np.asarray(node_type)
    
    x = pos_np[:, 0]
    y = pos_np[:, 1]
    
    # Normalized spatial coordinates in [-1, 1]
    x_span = float(x.max() - x.min())
    y_span = float(y.max() - y.min())
    xn = 2.0 * (x - x.min()) / (x_span if x_span > 1e-12 else 1.0) - 1.0
    yn = 2.0 * (y - y.min()) / (y_span if y_span > 1e-12 else 1.0) - 1.0
    
    is_fix_x = (node_type_np[:, 1] == 1)
    is_fix_y = (node_type_np[:, 2] == 1)
    free_x = (~is_fix_x).astype(np.float64)
    free_y = (~is_fix_y).astype(np.float64)
    
    V_raw = []
    # Generate polynomial monomials x^p y^q for degree p + q <= order
    for deg in range(order + 1):
        for p in range(deg + 1):
            q = deg - p
            poly = (xn ** p) * (yn ** q)
            vx = poly * free_x
            vy = poly * free_y
            
            # Virtual field with displacement only in x
            if np.linalg.norm(vx) > 1e-12:
                field_x = np.stack([vx, np.zeros_like(x)], axis=-1).reshape(-1)
                V_raw.append(field_x)
                
            # Virtual field with displacement only in y
            if np.linalg.norm(vy) > 1e-12:
                field_y = np.stack([np.zeros_like(x), vy], axis=-1).reshape(-1)
                V_raw.append(field_y)
                
    if len(V_raw) == 0:
        raise ValueError("No admissible virtual fields could be constructed with given boundary conditions.")
        
    V_mat = np.array(V_raw)  # (M_raw, 2 * N_nodes)
    
    # Orthonormalize via QR decomposition
    Q, R = np.linalg.qr(V_mat.T)  # Q is (2 * N_nodes, rank), orthonormal columns
    tol = 1e-10 * np.max(np.abs(np.diag(R)))
    rank = int(np.sum(np.abs(np.diag(R)) > tol))
    Q = Q[:, :rank]
    
    # Reshape back to (M, N_nodes, 2)
    V_basis = Q.T.reshape(rank, len(x), 2)
    return jnp.array(V_basis, dtype=jnp.float64)


def compute_global_virtual_residuals(
    R_nodes: jnp.ndarray,
    V_basis: jnp.ndarray
) -> jnp.ndarray:
    """
    Projects nodal residual force vectors onto the global virtual field basis:
    R_global[..., m] = sum_{i, d} V_basis[m, i, d] * R_nodes[..., i, d]
    
    Args:
        R_nodes: (..., N_nodes, 2) nodal residual vectors.
        V_basis: (M, N_nodes, 2) virtual field basis tensors.
        
    Returns:
        R_global: (..., M) global virtual work equilibrium residuals.
    """
    return jnp.einsum("mid,...id->...m", V_basis, R_nodes)

