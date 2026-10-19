"""
Smooth Ogden deviatoric energy for plane problems (JAX and PyTorch).

    Psi_dev = sum_k mu_k / alpha_k (lam1_bar^a_k + lam2_bar^a_k + lam3_bar^a_k - 3),   lam_bar = J^(-1/3) lam

The principal stretches come from C = F^T F with F block-diagonal (in-plane 2x2 block, out-of-plane F33: plane strain
F33 = 1, plane stress F33 = lambda3). Individual eigenvalues are not differentiable where they coincide (F = I, uniform
in-plane stretch), but the symmetric sum is. For the in-plane block with eigenvalues m +- sqrt(q),
    g(m + sqrt q) + g(m - sqrt q) = 2 g(m) + g''(m) q + g''''(m) q^2 / 12 + O(q^3),   g(x) = x^(alpha/2),
which is used for small q (double-where, so gradients stay finite); first and second derivatives (FEM tangent) are
then well defined everywhere.

alpha -> 0 is harmless: (lam_bar^a - 1)/a -> ln lam_bar, and sum_i ln lam_bar_i = 0, so a term with alpha = 0 (e.g. a
pruned exponent) contributes nothing. alpha is floored at 1e-6 in magnitude to avoid 0/0.
"""
import jax.numpy as jnp

_Q_TOL = 1e-10      # switch to the series below this (q / m^2)
_A_MIN = 1e-6


def _pair_sum(m, q, beta, xp, where):
    """g(m + sqrt q) + g(m - sqrt q) with g(x) = x^beta, smooth in q at q = 0."""
    small = q < _Q_TOL * m ** 2
    q_safe = where(small, m ** 2, q)                       # exact branch never sees q ~ 0
    d = xp.sqrt(q_safe)
    exact = xp.clip(m + d, 1e-12, None) ** beta + xp.clip(m - d, 1e-12, None) ** beta
    q_ser = where(small, q, 0.0)
    g2 = beta * (beta - 1.0) * m ** (beta - 2.0)
    g4 = beta * (beta - 1.0) * (beta - 2.0) * (beta - 3.0) * m ** (beta - 4.0)
    series = 2.0 * m ** beta + g2 * q_ser + g4 * q_ser ** 2 / 12.0
    return where(small, series, exact)


def ogden_dev_energy_jax(F, mus, alphas):
    """F: (2, 2) or (3, 3) block-diagonal deformation gradient; mus, alphas: (K,) arrays. Returns a scalar."""
    if F.shape[-2:] == (2, 2):
        F2, f33 = F, 1.0
    else:
        F2, f33 = F[:2, :2], F[2, 2]
    C = F2.T @ F2
    J = jnp.abs(jnp.linalg.det(F2) * f33)
    m = 0.5 * (C[0, 0] + C[1, 1])
    q = 0.25 * (C[0, 0] - C[1, 1]) ** 2 + C[0, 1] ** 2
    c33 = f33 ** 2
    energy = 0.0
    for k in range(mus.shape[0]):
        a = alphas[k]
        a_safe = jnp.where(jnp.abs(a) < _A_MIN, _A_MIN, a)
        s = _pair_sum(m, q, a_safe / 2.0, jnp, jnp.where) + c33 ** (a_safe / 2.0)
        energy = energy + mus[k] / a_safe * (J ** (-a_safe / 3.0) * s - 3.0)
    return energy


def ogden_dev_energy_torch(F, mus, alphas):
    """
    F: (N, 3, 3) or (N, 2, 2) batch; mus, alphas: (K,) tensors. Returns (N,). For the distillation, which needs
    derivatives with respect to the parameters only (F is data), so the stretches are taken one by one and each term is
    written as mu sum_i expm1(alpha t_i) / alpha with t_i = ln lam_bar_i: accurate in float32 also for alpha -> 0
    (limit sum_i t_i = 0), where mu/alpha (sum lam_bar^alpha - 3) would amplify rounding by 1/alpha.
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
        a = alphas[k]
        small = torch.abs(a) < _A_MIN
        a_safe = torch.where(small, torch.full_like(a, _A_MIN), a)
        terms = torch.where(small, t, torch.expm1(a_safe * t) / a_safe)            # (lam_bar^a - 1) / a
        energy = energy + mus[k] * terms.sum(-1)
    return energy


# --- position of the Ogden terms in the full distilled parameter vector -------------------------------------
# [C10..C03, E (10) | D1..D3 (3) | anisotropic (6) | mu1, alpha1, mu2, alpha2, mu3, alpha3 (6)] = 25
OGDEN_SLICE = slice(19, 25)
N_FULL = 25


def ogden_terms_from_vector(p):
    """(mus, alphas) of the Ogden block of a full parameter vector (zeros if the vector is shorter)."""
    o = p[OGDEN_SLICE] if p.shape[-1] >= N_FULL else jnp.zeros(6)
    return o[0::2], o[1::2]
