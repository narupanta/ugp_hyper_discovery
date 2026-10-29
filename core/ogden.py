"""
Smooth Ogden deviatoric energy for plane problems (JAX and PyTorch), standard (Abaqus) convention:

    Psi_dev = sum_k 2 mu_k / alpha_k^2 (lam1_bar^a_k + lam2_bar^a_k + lam3_bar^a_k - 3),   lam_bar = J^(-1/3) lam

mu_k > 0 is the term's contribution to the small-strain shear modulus (mu_0 = sum_k mu_k) for either sign of alpha_k,
so every term with mu_k > 0 is stable; alpha_k is real (negative exponents allowed).

Evaluation: with t_i = ln lam_bar_i and sum_i t_i = 0,
    2 mu / alpha^2 (sum_i lam_bar_i^alpha - 3) = 2 mu sum_i phi_alpha(t_i),   phi_alpha(t) = (exp(alpha t) - 1 - alpha t) / alpha^2,
which has no cancellation for small |alpha| (series t^2/2 + alpha t^3/6 + alpha^2 t^4/24 below |alpha| < 1e-3) and the
finite limit mu sum_i t_i^2 at alpha = 0 (e.g. a pruned exponent).

The principal stretches come from C = F^T F with F block-diagonal (in-plane 2x2 block, out-of-plane F33: plane strain
F33 = 1, plane stress F33 = lambda3). Individual eigenvalues are not differentiable where they coincide (F = I, uniform
in-plane stretch), but the symmetric sum over the in-plane pair is. For the in-plane eigenvalues m +- sqrt(q) of C,
    g(m + sqrt q) + g(m - sqrt q) = 2 g(m) + g2(m) q + g4(m) q^2 / 12 + O(q^3)     (g2, g4: 2nd and 4th derivatives),
which is used for small q (double-where, so gradients stay finite); first and second derivatives (FEM tangent) are
then well defined everywhere.
"""
import jax
import jax.numpy as jnp

_Q_TOL = 1e-10      # switch to the series below this (q / m^2)
_A_SMALL = 1e-3     # |alpha| below this: Taylor series of phi_alpha


def _phi(t, a, xp, where):
    """(exp(a t) - 1 - a t) / a^2, smooth in a (series for small |a|)."""
    small = xp.abs(a) < _A_SMALL
    a_safe = where(small, xp.ones_like(a), a)
    exact = (xp.expm1(a_safe * t) - a_safe * t) / a_safe ** 2
    series = t ** 2 / 2.0 + a * t ** 3 / 6.0 + a ** 2 * t ** 4 / 24.0
    return where(small, series, exact)


def _pair_sum_jax(g, m, q):
    """g(m + sqrt q) + g(m - sqrt q), smooth in q at q = 0 (series in q from the derivatives of the scalar g)."""
    small = q < _Q_TOL * m ** 2
    q_safe = jnp.where(small, m ** 2, q)                   # exact branch never sees q ~ 0
    d = jnp.sqrt(q_safe)
    exact = g(jnp.clip(m + d, 1e-12, None)) + g(jnp.clip(m - d, 1e-12, None))
    q_ser = jnp.where(small, q, 0.0)
    g2 = jax.grad(jax.grad(g))
    g4 = jax.grad(jax.grad(g2))
    series = 2.0 * g(m) + g2(m) * q_ser + g4(m) * q_ser ** 2 / 12.0
    return jnp.where(small, series, exact)


def ogden_dev_energy_jax(F, mus, alphas):
    """F: (2, 2) or (3, 3) block-diagonal deformation gradient; mus, alphas: (K,) arrays. Returns a scalar."""
    if F.shape[-2:] == (2, 2):
        F2, f33 = F, 1.0
    else:
        F2, f33 = F[:2, :2], F[2, 2]
    C = F2.T @ F2
    log_j3 = jnp.log(jnp.abs(jnp.linalg.det(F2) * f33)) / 3.0
    m = 0.5 * (C[0, 0] + C[1, 1])
    q = 0.25 * (C[0, 0] - C[1, 1]) ** 2 + C[0, 1] ** 2
    t3 = jnp.log(jnp.abs(f33)) - log_j3
    energy = 0.0
    for k in range(mus.shape[0]):
        a = alphas[k]
        g = lambda x, a=a: _phi(0.5 * jnp.log(x) - log_j3, a, jnp, jnp.where)    # x: eigenvalue of C (= lam^2)
        energy = energy + 2.0 * mus[k] * (_pair_sum_jax(g, m, q) + _phi(t3, a, jnp, jnp.where))
    return energy


def ogden_dev_energy_torch(F, mus, alphas):
    """
    F: (N, 3, 3) or (N, 2, 2) batch; mus, alphas: (K,) tensors. Returns (N,). For the distillation, which needs
    derivatives with respect to the parameters only (F is data), so the stretches are taken one by one (no smoothing at
    coincident stretches needed).
    """
    import torch
    if F.shape[-2:] == (2, 2):
        F2, f33 = F, torch.ones(F.shape[0], dtype=F.dtype, device=F.device)
    else:
        F2, f33 = F[:, :2, :2], F[:, 2, 2]
    C = F2.transpose(1, 2) @ F2
    J = torch.abs(torch.det(F2) * f33)
    m = 0.5 * (C[:, 0, 0] + C[:, 1, 1])
    d = torch.sqrt(torch.clamp(0.25 * (C[:, 0, 0] - C[:, 1, 1]) ** 2 + C[:, 0, 1] ** 2, min=0.0))
    log_j = torch.log(J) / 3.0
    t = torch.stack([0.5 * torch.log(torch.clamp(m + d, min=1e-12)) - log_j,      # ln lam_bar_i, i = 1, 2, 3
                     0.5 * torch.log(torch.clamp(m - d, min=1e-12)) - log_j,
                     torch.log(torch.abs(f33)) - log_j], dim=-1)                       # (N, 3)
    energy = torch.zeros_like(m)
    for k in range(mus.shape[0]):
        energy = energy + 2.0 * mus[k] * _phi(t, alphas[k], torch, torch.where).sum(-1)
    return energy


# --- position of the Ogden terms in the full distilled parameter vector -------------------------------------
# [C10..C03, E (10) | D1..D3 (3) | anisotropic (6) | mu1, alpha1, ..., mu8, alpha8 (16)] = 35 (core/material_library.py)
from core.material_library import N_FULL, OGDEN_NAMES, OGDEN_START  # noqa: E402
OGDEN_SLICE = slice(OGDEN_START, OGDEN_START + len(OGDEN_NAMES))


def ogden_terms_from_vector(p):
    """(mus, alphas) of the Ogden block of a full parameter vector (zeros if the vector is shorter)."""
    o = p[OGDEN_SLICE] if p.shape[-1] >= N_FULL else jnp.zeros(len(OGDEN_NAMES))
    return o[0::2], o[1::2]
