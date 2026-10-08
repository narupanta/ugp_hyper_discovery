"""
Reduced-rank (Hilbert-space) GP basis for the HSGP extraction method (Solin & Sarkka 2020, Stat. Comput. 30:419).

A stationary GP f ~ GP(0, k) on a box [c - B, c + B]^d is expanded in the Dirichlet Laplacian eigenfunctions,
    f(x) ~= sum_k phi_k(x) w_k,  w_k ~ N(0, S(sqrt(lambda_k))),
with S the spectral density of k. The basis depends only on the box; the hyperparameters (sigma, lengthscales)
enter only through the prior variances S, so quantities assembled from the basis (e.g. FE internal forces) are
computed once. The approximation is accurate inside the box away from its edges (all functions vanish at the edges).
"""
from typing import NamedTuple

import itertools
import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


class Box(NamedTuple):
    """Tensor-product HSGP basis on [centre - half_width, centre + half_width]^d."""
    centre: np.ndarray      # (d,)
    half_width: np.ndarray  # (d,)
    indices: np.ndarray     # (K, d) eigenfunction multi-indices, entries 1..m_i

    @property
    def dim(self) -> int:
        return int(self.centre.shape[0])

    @property
    def num_basis(self) -> int:
        return int(self.indices.shape[0])

    def sqrt_eigenvalues(self) -> np.ndarray:
        """(K, d) frequencies pi j / (2 B) per dimension."""
        return np.pi * self.indices / (2.0 * self.half_width[None, :])


def make_box(points: np.ndarray, reference: np.ndarray, box_factor: float, num_per_dim) -> Box:
    """
    Box around the points and the reference: half-width = box_factor * half-range of their union.
    num_per_dim: int or (d,) number of eigenfunctions per dimension (full tensor product).
    """
    pts = np.concatenate([np.asarray(points, dtype=np.float64).reshape(-1, len(reference)),
                          np.asarray(reference, dtype=np.float64)[None, :]], axis=0)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    centre = 0.5 * (lo + hi)
    half_range = np.maximum(0.5 * (hi - lo), 1e-6)
    m = np.broadcast_to(np.asarray(num_per_dim, dtype=int), centre.shape)
    indices = np.array(list(itertools.product(*[range(1, mi + 1) for mi in m])), dtype=np.float64)
    return Box(centre=centre, half_width=box_factor * half_range, indices=indices)


def eigenfunctions(box: Box, x: jnp.ndarray) -> jnp.ndarray:
    """phi(x): (..., d) -> (..., K), prod_i sin(pi j_i (x_i - c_i + B_i) / (2 B_i)) / sqrt(B_i)."""
    c = jnp.asarray(box.centre)
    B = jnp.asarray(box.half_width)
    freq = jnp.asarray(box.sqrt_eigenvalues())                      # (K, d)
    arg = (x[..., None, :] - c + B) * freq                          # (..., K, d)
    return jnp.prod(jnp.sin(arg) / jnp.sqrt(B), axis=-1)


def se_spectral_density(box: Box, sigma: jnp.ndarray, lengthscales: jnp.ndarray) -> jnp.ndarray:
    """Prior variances S(sqrt(lambda_k)) of the ARD squared-exponential kernel sigma^2 exp(-|r/l|^2 / 2): (K,)."""
    w = jnp.asarray(box.sqrt_eigenvalues())
    ls = jnp.broadcast_to(lengthscales, (box.dim,))
    return sigma ** 2 * (2.0 * jnp.pi) ** (box.dim / 2.0) * jnp.prod(ls) * jnp.exp(-0.5 * jnp.sum((ls * w) ** 2, axis=-1))


def inside_interior(box: Box, x: np.ndarray, margin: np.ndarray) -> np.ndarray:
    """True where x lies at least `margin` (per dimension, e.g. one lengthscale) inside the box edges."""
    x = np.asarray(x).reshape(-1, box.dim)
    dist = box.half_width[None, :] - np.abs(x - box.centre[None, :])
    return np.all(dist >= np.asarray(margin)[None, :], axis=1)


def recommended_num_basis(box_factor: float, lengthscale_over_half_range: float) -> int:
    """Riutort-Mayol et al. (2023, Stat. Comput. 33:17) rule for the SE kernel: m >= 1.75 c / (l / S)."""
    return int(np.ceil(1.75 * box_factor / max(lengthscale_over_half_range, 1e-6)))
