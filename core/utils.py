import os
import jax
import jax.numpy as jnp
import numpy as np
from sklearn.neighbors import NearestNeighbors
# -------------------------------
# Tensor utility functions
# -------------------------------

def J_func(f):
    """Determinant of deformation gradient tensor."""
    return jnp.linalg.det(f)  # shape [N]

def B_func(f):
    """Left Cauchy-Green tensor."""
    return f @ jnp.swapaxes(f, -2, -1)  # [N,d,d]
def C_func(f) :
    return jnp.swapaxes(f, -2, -1) @ f  # [N,d,d]
def I1_func(B):
    """First invariant (trace)."""
    return jnp.trace(B, axis1=-2, axis2=-1)

def I2_func(B):
    """Second invariant."""
    trB = jnp.trace(B, axis1=-2, axis2=-1)
    trBB = jnp.trace(B @ B, axis1=-2, axis2=-1)
    return 0.5 * (trB**2 - trBB)

def I3_func(B):
    """Third invariant (determinant)."""
    return jnp.linalg.det(B)

def I4_func(C, a0):
    """Fourth invariant (pseudo-invariant) I4 = a0 . C a0."""
    return jnp.einsum('i,...ij,j->...', a0, C, a0)

def I5_func(C, a0):
    """Fifth invariant (pseudo-invariant) I5 = a0 . C^2 a0."""
    C2 = jnp.einsum('...ij,...jk->...ik', C, C)
    return jnp.einsum('i,...ij,j->...', a0, C2, a0)

# -------------------------------
# Strain energy functions
# -------------------------------


# -------------------------------
# Solve for coefficients (batched)
# -------------------------------

def solve_for_coefficients_batched(lambda_B, lambda_sigma):
    """
    Batched version of solve_for_coefficients.

    Args:
        lambda_B: array (batch_size, 3)
        lambda_sigma: array (batch_size, 3)

    Returns:
        coefficients_c: (batch_size, 3)
        V: (batch_size, 3, 3)
    """
    if lambda_B.ndim != 2 or lambda_B.shape[1] != 3:
        raise ValueError("lambda_B must have shape (batch_size, 3)")
    if lambda_sigma.ndim != 2 or lambda_sigma.shape[1] != 3:
        raise ValueError("lambda_sigma must have shape (batch_size, 3)")
    if lambda_B.shape[0] != lambda_sigma.shape[0]:
        raise ValueError("Batch sizes of lambda_B and lambda_sigma must match.")

    # Construct Vandermonde matrices
    col1 = jnp.ones_like(lambda_B)
    col2 = lambda_B
    col3 = lambda_B**2
    V = jnp.stack((col1, col2, col3), axis=-1)  # (batch_size, 3, 3)

    # Compute pseudoinverse and solve for coefficients
    V_pinv = jnp.linalg.pinv(V)
    coefficients_c = jnp.einsum("bij,bj->bi", V_pinv, lambda_sigma)
    return coefficients_c, V


from jax import vmap

def fto3x3(f) :
    f3x3 = jnp.array([[f[0,0], f[0,1], 0.0],
                      [f[1,0], f[1,1], 0.0],
                      [0.0, 0.0, 1.0]])
    return f3x3

# def fto3x3(f):
#     # Initialize a 3x3 Identity matrix (ensures F33 = 1.0 and others are 0.0)
#     f3x3 = jnp.eye(3)
#     # Use JAX's functional update to place the 2x2 F into the 3x3
#     f3x3 = f3x3.at[:2, :2].set(f)
#     return f3x3

@jax.vmap
def transformation_jacobian(coords_elem) :
    x1, y1 = coords_elem[0]
    x2, y2 = coords_elem[1]
    x3, y3 = coords_elem[2]

    # Jacobian of shape function derivatives
    J = jnp.array([
        [x2 - x1, y2 - y1],
        [x3 - x1, y3 - y1]
    ])
    return J

@jax.vmap
def deformation_gradient_element(coords_elem, disp_elem):
    x1, y1 = coords_elem[0]
    x2, y2 = coords_elem[1]
    x3, y3 = coords_elem[2]

    # Jacobian of shape function derivatives
    J = jnp.array([
        [x2 - x1, y2 - y1],
        [x3 - x1, y3 - y1]
    ])

    # Area factor
    detJ = jnp.linalg.det(J)

    # Shape function derivatives in reference space
    dN_ref = jnp.array([
        [-1., -1.],
        [ 1.,  0.],
        [ 0.,  1.]
    ])

    # Convert to physical derivatives: dN/dx = inv(J)^T * dN_ref
    dNdx = jnp.transpose(jnp.linalg.solve(J, dN_ref.T))

    # Gradient of displacement
    gradu = disp_elem.T @ dNdx  # 2x3 @ 3x2 = 2x2

    # Deformation gradient
    F = jnp.eye(2) + gradu
    return F, dNdx


def calculate_min_ls(z):
    # For a 2D/3D point cloud, a quick way is to use the 
    # average distance to the nearest neighbor.
    nbrs = NearestNeighbors(n_neighbors=2).fit(z)
    distances, _ = nbrs.kneighbors(z)
    avg_dist = jnp.mean(distances[:, 1])
    return avg_dist * 0.5 # Minimum allowable lengthscale

def invariants_and_derivatives(f):
    # Calculate invariants
    C = f.T @ f
    I1 = jnp.trace(C)
    I2 = 0.5 * (I1**2 - jnp.trace(C @ C))
    I3 = jnp.linalg.det(C)
    # derivatives wrt F (2x2)
    dI1_dF = 2*f
    dI2_dF = 2*(I1*f - f @ C)
    dI3_dF = 2*jnp.linalg.det(f)**2 * jnp.linalg.inv(f).T
    dI_dF = jnp.stack([dI1_dF, dI2_dF, dI3_dF])  # (3,2,2)
    return jnp.array([I1, I2, I3]), dI_dF


def farthest_point_sampling(pts, num_samples):
    """
    pts: (N, 3) array of points
    num_samples: 25
    """
    n_pts = pts.shape[0]
    # Initialize: pick the first point in the list as the start
    selected_indices = jnp.zeros(num_samples, dtype=jnp.int32)
    
    # Track the distance from every point to its NEAREST selected point
    # Start with infinity
    dist_to_set = jnp.full((n_pts,), jnp.inf)
    
    def scan_body(dist_to_set, i):
        # The next point is the one farthest from the current set
        idx = jnp.argmax(dist_to_set)
        
        # Calculate distance from the new point to all other points
        new_pt = pts[idx]
        dists = jnp.sum((pts - new_pt)**2, axis=-1) # Squared Euclidean
        
        # Update distances: dist to set is min(old_dist, dist_to_new_point)
        dist_to_set = jnp.minimum(dist_to_set, dists)
        
        return dist_to_set, idx

    # We manually pick the first point to start
    first_idx = 0
    dist_to_set = jnp.sum((pts - pts[first_idx])**2, axis=-1)
    
    # Run the loop for the remaining 24 points
    _, remaining_indices = jax.lax.scan(scan_body, dist_to_set, jnp.arange(1, num_samples))
    
    return jnp.concatenate([jnp.array([first_idx]), remaining_indices])

def transform_input_features(invariants) :
    i3 = jnp.maximum(invariants[2], 1e-6)
    j = jnp.sqrt(i3)
    i1_dev = i3**(-1/3)*invariants[0]
    i2_dev = i3**(-2/3)*invariants[1]
    dev_feature = jnp.stack([i1_dev, i2_dev], axis = -1)
    vol_feature = jnp.array([j])

    return dev_feature, vol_feature

def farthest_point_sampling_with_fixed_point(pts, num_samples, fixed_point):
    """
    pts: (N, 2) array of 2D points
    num_samples: Total points to return (including the fixed point)
    fixed_point: jnp.array([3.0, 3.0])
    """
    # 1. Add the fixed point [3,3] to the very start of the array
    # This guarantees it exists in the data we are sampling from
    pts_augmented = jnp.concatenate([fixed_point[None, :], pts], axis=0)
    
    # 2. The index of [3,3] is now 0
    first_idx = 0
    
    # 3. Initialize distance to set based on the fixed point [3,3]
    initial_dist_to_set = jnp.sum((pts_augmented - pts_augmented[first_idx])**2, axis=-1)
    
    def scan_body(dist_to_set, _):
        # Pick point farthest from the current set (initially just [3,3])
        idx = jnp.argmax(dist_to_set)
        
        new_pt = pts_augmented[idx]
        dists_to_new_pt = jnp.sum((pts_augmented - new_pt)**2, axis=-1)
        
        # Update distances: nearest distance to the set
        new_dist_to_set = jnp.minimum(dist_to_set, dists_to_new_pt)
        
        return new_dist_to_set, idx

    # 4. Run the loop for the remaining points
    _, remaining_indices = jax.lax.scan(
        scan_body, 
        initial_dist_to_set, 
        None, 
        length=num_samples - 1
    )
    
    sampled_indices = jnp.concatenate([jnp.array([first_idx]), remaining_indices])
    
    # Return the coordinates of the sampled points
    return pts_augmented[sampled_indices]

def infer_material_model_name(path: str) -> str:
    """
    Infers the material model name from the saved_model_dir path or its metadata/config.
    Raises ValueError if the model name cannot be determined.
    """
    if not path:
        raise ValueError("Cannot infer material model name: path is empty or None.")

    try:
        cfg = load_model_config(path)
        if "material_model_name" in cfg and cfg["material_model_name"]:
            return str(cfg["material_model_name"])
    except Exception:
        pass

    known_models = [
        "full_isotropic", "full_iso", "iso_full", "poly_iso",
        "symnonortho60", "neohookean4", "neohookean2", "c20_d10_d05", "c20d10d05",
        "gentthomas", "ortho45", "ortho090", "ortho900", "aniso30", "aniso45", "isihara", "nh4", "nh2", "nh"
    ]
    abs_path = os.path.abspath(path)
    # Check optimization_log.txt first if it exists in the directory, parent, or extraction subfolder
    for check_dir in [abs_path, os.path.dirname(abs_path), os.path.join(abs_path, "extraction")]:
        log_file = os.path.join(check_dir, "optimization_log.txt")
        if os.path.exists(log_file):
            try:
                with open(log_file, "r", encoding="utf-8") as f:
                    first_line = f.readline()
                    log_tokens = [t.strip().strip(",[]") for t in first_line.split()]
                    for model_cand in known_models:
                        if model_cand in log_tokens or model_cand == first_line.split(",")[-1].strip():
                            return model_cand
            except Exception:
                pass
    # Traverse path components from leaf to root
    for p in reversed(abs_path.split(os.sep)):
        subparts = p.split('_')
        for model_cand in known_models:
            if model_cand in subparts or model_cand == p:
                return model_cand
                
    raise ValueError(f"Could not infer material model name from path '{path}'. Please ensure the path contains a valid model name or optimization_log.txt.")


def load_model_config(path: str) -> dict:
    """
    Loads model configuration dictionary from a saved model directory or parent/subfolder.
    Checks config.json, config.yaml, and recipe_config.yaml.
    Raises FileNotFoundError or ValueError if no configuration is found.
    """
    if not path:
        raise ValueError("Cannot load model config: path is empty or None.")
    
    import json
    import yaml
    
    abs_path = os.path.abspath(path)
    search_dirs = [
        abs_path,
        os.path.dirname(abs_path),
        os.path.join(abs_path, "extraction"),
        os.path.join(abs_path, "..")
    ]
    
    # Check source_extraction_dir if present in distillation folder
    for s_file in ["dev_source_extraction_dir.txt", "vol_source_extraction_dir.txt", "aniso_source_extraction_dir.txt", "source_extraction_dir.txt"]:
        for d in search_dirs[:2]:
            sf_path = os.path.join(d, s_file)
            if os.path.exists(sf_path):
                try:
                    with open(sf_path, "r") as f:
                        src_dir = f.read().strip()
                        if os.path.isdir(src_dir):
                            search_dirs.append(os.path.abspath(src_dir))
                except Exception:
                    pass

    for d in search_dirs:
        for cfg_name in ["recipe_config.yaml", "config.json", "config.yaml"]:
            cfg_path = os.path.join(d, cfg_name)
            if os.path.exists(cfg_path):
                try:
                    if cfg_name.endswith(".json"):
                        with open(cfg_path, "r") as f:
                            cfg = json.load(f)
                    else:
                        with open(cfg_path, "r") as f:
                            cfg = yaml.safe_load(f)
                    if isinstance(cfg, dict):
                        return cfg
                except Exception:
                    pass

    raise FileNotFoundError(f"No configuration file (config.json / config.yaml / recipe_config.yaml) found in or around '{path}'.")