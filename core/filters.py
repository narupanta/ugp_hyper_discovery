"""
core/filters.py: Displacement Noise Filtering & Preprocessing for FE Meshes

Provides mesh-aware spatial filtering routines (Laplacian diffusion, Gaussian kernel)
to remove measurement noise from full-field displacement data while strictly preserving
boundary conditions (Dirichlet clamps and traction boundaries) and recomputing
consistent kinematics and strain invariants.
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

from core.utils import deformation_gradient_element
from core.fem_engine import compute_all_invariants


def build_mesh_adjacency(
    cells: np.ndarray,
    mesh_pos: np.ndarray,
    weighted: bool = True
) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    """
    Constructs the node neighbor graph and optional distance-based diffusion weights
    for an unstructured triangular mesh.

    Args:
        cells: (n_cells, 3) element node connectivity.
        mesh_pos: (n_nodes, 2) nodal reference coordinates.
        weighted: If True, computes inverse-distance weights w_ij = 1 / ||X_i - X_j||.

    Returns:
        neighbors_dict: Mapping node_idx -> array of neighbor node indices.
        weights_dict: Mapping node_idx -> array of normalized weights summing to 1.0.
    """
    n_nodes = mesh_pos.shape[0]
    raw_adj = {i: set() for i in range(n_nodes)}

    cells_np = np.asarray(cells, dtype=np.int32)
    for c in cells_np:
        i, j, k = c[0], c[1], c[2]
        raw_adj[i].add(j); raw_adj[i].add(k)
        raw_adj[j].add(i); raw_adj[j].add(k)
        raw_adj[k].add(i); raw_adj[k].add(j)

    neighbors_dict = {}
    weights_dict = {}
    pos = np.asarray(mesh_pos[:, :2], dtype=np.float64)

    for i in range(n_nodes):
        nbrs = np.array(sorted(list(raw_adj[i])), dtype=np.int32)
        neighbors_dict[i] = nbrs
        if len(nbrs) == 0:
            weights_dict[i] = np.array([1.0], dtype=np.float64)
            continue

        if weighted:
            diffs = pos[nbrs] - pos[i]
            dists = np.linalg.norm(diffs, axis=1)
            # Safe inverse distance
            inv_dists = 1.0 / np.maximum(dists, 1e-12)
            weights = inv_dists / np.sum(inv_dists)
        else:
            weights = np.full(len(nbrs), 1.0 / len(nbrs), dtype=np.float64)

        weights_dict[i] = weights

    return neighbors_dict, weights_dict


def smooth_displacements_laplacian(
    u: np.ndarray,
    mesh_pos: np.ndarray,
    cells: np.ndarray,
    node_type: np.ndarray,
    alpha: float = 0.5,
    passes: int = 3,
    weighted: bool = True
) -> np.ndarray:
    """
    Applies distance-weighted Laplacian diffusion to smooth full-field displacement data,
    strictly pinning fixed boundary nodes (Dirichlet and monitored traction borders).

    Args:
        u: (n_steps, n_nodes, 2) or (n_nodes, 2) displacement array.
        mesh_pos: (n_nodes, 2) nodal reference coordinates.
        cells: (n_cells, 3) element connectivity.
        node_type: (n_nodes, 3) boundary condition flags.
        alpha: Smoothing relaxation rate in (0, 1]. Higher = stronger smoothing per pass.
        passes: Number of diffusion iterations.
        weighted: If True, uses inverse Euclidean distance weighting.

    Returns:
        u_smooth: Smoothed displacement array matching input shape.
    """
    orig_ndim = u.ndim
    if orig_ndim == 2:
        u_arr = np.array(u)[None, ...]
    else:
        u_arr = np.array(u)

    num_steps, n_nodes, dim = u_arr.shape
    node_type_np = np.asarray(node_type)

    # Free nodes where displacements are unconstrained in both X and Y
    free_nodes = (node_type_np[:, 1] != 1) & (node_type_np[:, 2] != 1)
    free_indices = np.where(free_nodes)[0]

    neighbors_dict, weights_dict = build_mesh_adjacency(cells, mesh_pos, weighted=weighted)

    u_smooth = np.copy(u_arr)

    for p in range(passes):
        u_next = np.copy(u_smooth)
        for i in free_indices:
            nbrs = neighbors_dict[i]
            w = weights_dict[i]
            # Weighted neighbor average across all load steps and displacement components: (num_steps, dim)
            nbr_vals = u_smooth[:, nbrs, :]  # (num_steps, k, dim)
            nbr_mean = np.einsum('k,skd->sd', w, nbr_vals)

            u_next[:, i, :] = (1.0 - alpha) * u_smooth[:, i, :] + alpha * nbr_mean

        u_smooth = u_next

    # Ensure fixed nodes strictly match original values
    u_smooth[:, ~free_nodes, :] = u_arr[:, ~free_nodes, :]

    if orig_ndim == 2:
        return u_smooth[0]
    return u_smooth


def smooth_displacements_gaussian(
    u: np.ndarray,
    mesh_pos: np.ndarray,
    node_type: np.ndarray,
    radius: Optional[float] = None,
    sigma: Optional[float] = None
) -> np.ndarray:
    """
    Applies a spatial Gaussian kernel filter to smooth displacement measurements,
    strictly preserving boundary nodes.
    """
    orig_ndim = u.ndim
    if orig_ndim == 2:
        u_arr = np.array(u)[None, ...]
    else:
        u_arr = np.array(u)

    num_steps, n_nodes, dim = u_arr.shape
    pos = np.asarray(mesh_pos[:, :2], dtype=np.float64)
    node_type_np = np.asarray(node_type)
    free_nodes = (node_type_np[:, 1] != 1) & (node_type_np[:, 2] != 1)
    free_indices = np.where(free_nodes)[0]

    # Heuristic default radius: 2.5 * median nearest neighbor distance
    if radius is None or sigma is None:
        diffs = pos[:, None, :] - pos[None, :, :]
        dists = np.linalg.norm(diffs, axis=-1)
        np.fill_diagonal(dists, np.inf)
        min_dists = np.min(dists, axis=1)
        med_dist = float(np.median(min_dists))
        if radius is None:
            radius = 2.5 * med_dist
        if sigma is None:
            sigma = radius / 2.0

    u_smooth = np.copy(u_arr)
    diffs = pos[:, None, :] - pos[None, :, :]
    dists_sq = np.sum(diffs**2, axis=-1)

    for i in free_indices:
        mask = (dists_sq[i] <= radius**2)
        nbr_idx = np.where(mask)[0]
        if len(nbr_idx) == 0:
            continue
        d_sq = dists_sq[i, nbr_idx]
        w = np.exp(-d_sq / (2.0 * sigma**2))
        w /= np.sum(w)

        nbr_vals = u_arr[:, nbr_idx, :]  # (num_steps, k, dim)
        u_smooth[:, i, :] = np.einsum('k,skd->sd', w, nbr_vals)

    u_smooth[:, ~free_nodes, :] = u_arr[:, ~free_nodes, :]

    if orig_ndim == 2:
        return u_smooth[0]
    return u_smooth


def filter_dataset_kinematics(
    prep_data: Dict[str, Any],
    method: str = "laplacian",
    alpha: float = 0.5,
    passes: int = 3,
    radius: Optional[float] = None,
    sigma: Optional[float] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Comprehensive dataset preprocessor that filters displacement noise,
    recomputes element deformation gradients (F) and strain invariants,
    while preserving original raw data under 'u_raw', 'F_raw', and 'raw_*_bar'.

    Args:
        prep_data: Dataset dictionary from PrecomputedVFMDataset or synthetic exporter.
        method: 'laplacian' or 'gaussian'.
        alpha: Laplacian diffusion relaxation rate (default 0.5).
        passes: Laplacian diffusion passes (default 3).
        radius: Gaussian kernel cutoff radius.
        sigma: Gaussian kernel bandwidth.
        verbose: If True, prints filtering summary and noise reduction metrics.

    Returns:
        prep_data: Updated dictionary containing smoothed kinematics and preserved raw data.
    """
    mesh_pos = np.asarray(prep_data["mesh_pos"], dtype=np.float64)
    cells = np.asarray(prep_data["cells"], dtype=np.int32)
    node_type = np.asarray(prep_data["node_type"])

    # Locate observed displacement field
    u_source = prep_data.get("u_obs", prep_data.get("u"))
    if u_source is None:
        raise ValueError("Cannot filter dataset: neither 'u_obs' nor 'u' found in prep_data.")

    u_raw = np.asarray(u_source, dtype=np.float64)
    num_steps = u_raw.shape[0]

    # Preserve raw data under dedicated keys if not already preserved
    if "u_raw" not in prep_data:
        prep_data["u_raw"] = u_raw
    if "F_raw" not in prep_data and "F" in prep_data:
        prep_data["F_raw"] = np.asarray(prep_data["F"])
    if "raw_I1_bar" not in prep_data and "obs_I1_bar" in prep_data:
        prep_data["raw_I1_bar"] = np.asarray(prep_data["obs_I1_bar"])
    if "raw_I2_bar" not in prep_data and "obs_I2_bar" in prep_data:
        prep_data["raw_I2_bar"] = np.asarray(prep_data["obs_I2_bar"])
    if "raw_J" not in prep_data and "obs_J" in prep_data:
        prep_data["raw_J"] = np.asarray(prep_data["obs_J"])
    if "raw_I4_bar" not in prep_data and "obs_I4_bar" in prep_data:
        prep_data["raw_I4_bar"] = np.asarray(prep_data["obs_I4_bar"])
    if "raw_I6_bar" not in prep_data and "obs_I6_bar" in prep_data:
        prep_data["raw_I6_bar"] = np.asarray(prep_data["obs_I6_bar"])
    if "raw_I8_bar" not in prep_data and "obs_I8_bar" in prep_data:
        prep_data["raw_I8_bar"] = np.asarray(prep_data["obs_I8_bar"])

    # 1. Apply displacement smoothing
    if method.lower() == "gaussian":
        u_smooth = smooth_displacements_gaussian(
            u_raw, mesh_pos, node_type, radius=radius, sigma=sigma
        )
    else:
        u_smooth = smooth_displacements_laplacian(
            u_raw, mesh_pos, cells, node_type, alpha=alpha, passes=passes, weighted=True
        )

    # 2. Recompute deformation gradients: F = I + gradu
    m_cells = jnp.asarray(mesh_pos[cells, :2])  # (n_cells, 3, 2)
    u_cells_smooth = jnp.asarray(u_smooth[:, cells, :])  # (n_steps, n_cells, 3, 2)

    # Vectorize across load steps using deformation_gradient_element
    recompute_step_fn = jax.vmap(lambda uc: deformation_gradient_element(m_cells, uc)[0])
    F_smooth = np.array(recompute_step_fn(u_cells_smooth))  # (n_steps, n_cells, 2, 2)

    # 3. Recompute strain invariants
    a0 = prep_data.get("a0", None)
    a1 = prep_data.get("a1", None)
    if hasattr(a0, "dtype") and a0.dtype == object and getattr(a0, "ndim", 0) == 0:
        a0 = a0.item()
    if hasattr(a1, "dtype") and a1.dtype == object and getattr(a1, "ndim", 0) == 0:
        a1 = a1.item()

    inv_smooth = compute_all_invariants(jnp.asarray(F_smooth), a0=a0, a1=a1)

    # 4. Update dataset dictionary with filtered fields
    prep_data["u"] = u_smooth
    prep_data["u_obs"] = u_smooth
    prep_data["u_filt"] = u_smooth

    prep_data["F"] = F_smooth
    prep_data["F_obs"] = F_smooth
    prep_data["F_filt"] = F_smooth

    prep_data["obs_I1_bar"] = np.asarray(inv_smooth["I1_bar"])
    prep_data["obs_I2_bar"] = np.asarray(inv_smooth["I2_bar"])
    prep_data["obs_J"] = np.asarray(inv_smooth["J"])

    if "I4_bar" in inv_smooth:
        prep_data["obs_I4_bar"] = np.asarray(inv_smooth["I4_bar"])
    if "I6_bar" in inv_smooth:
        prep_data["obs_I6_bar"] = np.asarray(inv_smooth["I6_bar"])
    if "I8_bar" in inv_smooth:
        prep_data["obs_I8_bar"] = np.asarray(inv_smooth["I8_bar"])

    # Attach filter metadata
    filter_meta = {
        "is_filtered": True,
        "method": method,
        "alpha": alpha if method == "laplacian" else None,
        "passes": passes if method == "laplacian" else None,
        "radius": radius if method == "gaussian" else None,
        "sigma": sigma if method == "gaussian" else None,
    }
    prep_data["filter_metadata"] = filter_meta

    if verbose:
        u_diff = np.abs(u_smooth - u_raw)
        free_nodes = (node_type[:, 1] != 1) & (node_type[:, 2] != 1)
        mean_shift = np.mean(u_diff[:, free_nodes, :])
        max_shift = np.max(u_diff[:, free_nodes, :])

        u_true = prep_data.get("u_true", None)
        true_str = ""
        if u_true is not None:
            raw_err = np.mean(np.abs(u_raw - u_true)[:, free_nodes, :])
            filt_err = np.mean(np.abs(u_smooth - u_true)[:, free_nodes, :])
            reduction = (1.0 - filt_err / max(raw_err, 1e-12)) * 100.0
            true_str = f" | Error vs True u: raw={raw_err:.2e} -> filt={filt_err:.2e} ({reduction:+.1f}% noise reduction)"

        print(f"[FILTER] Applied {method} filter ({passes} passes, alpha={alpha}). Mean shift={mean_shift:.2e}, Max shift={max_shift:.2e}{true_str}")

    return prep_data

