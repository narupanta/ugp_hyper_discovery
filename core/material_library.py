"""
Candidate libraries of the distillation and the full parameter vector shared by the FEM validation and the plots.

Full vector layout (positions are fixed; shorter vectors mean the later blocks are zero):
    [C10, C01, C20, C11, C02, C30, C21, C12, C03, E | D1, D2, D3 | anisotropic (6) | mu1, alpha1, ..., mu8, alpha8 |
     0 ............................................ 9  10 ..... 12  13 ......... 18  19 ......................... 34
     C40, C31, C22, C13, C04, C50, ...]
     35 .........................   (GMR terms of order > 3, only with gmr_order > 3)

GMR order N (recipe gmr_order, default 3): the polynomial terms C_ij (I1_bar - 3)^i (I2_bar - 3)^j with 1 <= i + j <= N,
ordered by degree and decreasing i. The GMR parameter vector of order N is [C10 .. C03, E, C40, C31, ...]: the
third-order terms and E keep positions 0..9 for every N (an order below 3 leaves the missing terms at zero), so
dev_params_disabled indices and older runs keep their meaning; the terms of order 4..N follow from position 10.

Deviatoric libraries (the distilled dev samples): "gmr" polynomial terms (10 columns; the log term E is a column but is
only fitted for "gmr_log"), "ogden" N_OGDEN_TERMS Ogden terms (2 N columns: mu1, alpha1, ...), "gmr_ogden" both
(polynomial first). The volumetric samples are D1..D3. Older runs with 3 Ogden terms (6 columns) still load: the
missing terms are zero.
"""
import json
import os

import numpy as np

GMR_DEV_NAMES = ["C10", "C01", "C20", "C11", "C02", "C30", "C21", "C12", "C03", "E"]    # order 3 (base block)
GMR_BASE_ORDER = 3


def gmr_high_terms(order):
    """(i, j) exponents of the GMR terms of degree 4..order (after E in the layout), by degree and decreasing i."""
    return [(i, d - i) for d in range(GMR_BASE_ORDER + 1, int(order) + 1) for i in range(d, -1, -1)]


def gmr_terms(order=GMR_BASE_ORDER):
    """(i, j) exponents of every polynomial column of the GMR vector of this order, None for the log column E."""
    base = [(1, 0), (0, 1), (2, 0), (1, 1), (0, 2), (3, 0), (2, 1), (1, 2), (0, 3)]
    return base + [None] + gmr_high_terms(order)


def gmr_names(order=GMR_BASE_ORDER):
    return GMR_DEV_NAMES + [f"C{i}{j}" for i, j in gmr_high_terms(order)]


def n_gmr(order=GMR_BASE_ORDER):
    """Columns of the GMR parameter vector of this order (10 up to order 3)."""
    return len(GMR_DEV_NAMES) + len(gmr_high_terms(order))


def gmr_order_from_length(n):
    """Order of a GMR parameter vector of n columns (10 -> 3, 15 -> 4, 21 -> 5, ...)."""
    order = GMR_BASE_ORDER
    while n_gmr(order) < n:
        order += 1
    if n_gmr(order) != n:
        raise ValueError(f"{n} columns is not a GMR parameter vector of any order (10, 15, 21, 28, ...)")
    return order


def gmr_dev_energy(p, i1m3, i2m3, i2_bar, xp):
    """GMR deviatoric energy of the parameter vector p (any order) at I1_bar - 3, I2_bar - 3 (array module xp)."""
    w = 0.0
    for k, ij in enumerate(gmr_terms(gmr_order_from_length(len(p)))):
        if ij is None:
            w = w + p[k] * xp.log(xp.maximum(i2_bar / 3.0, 1e-8))
        else:
            w = w + p[k] * i1m3 ** ij[0] * i2m3 ** ij[1]
    return w
VOL_NAMES = ["D1", "D2", "D3"]
ANISO_NAMES = ["C42", "C43", "C44", "C62", "C63", "C64"]
N_OGDEN_TERMS = 8
# exponents the Ogden distillation starts from (stage 1 keeps them fixed and fits the mu_k only; core of the library)
OGDEN_ALPHA_INIT = [-4.0, -3.0, -2.0, -1.0, 1.0, 2.0, 3.0, 4.0]
OGDEN_NAMES = [n for k in range(1, N_OGDEN_TERMS + 1) for n in (f"mu{k}", f"alpha{k}")]
FULL_NAMES = GMR_DEV_NAMES + VOL_NAMES + ANISO_NAMES + OGDEN_NAMES      # 35
OGDEN_START = 19
N_FULL = len(FULL_NAMES)                 # base full vector; GMR terms of order > 3 follow at HIGH_START
HIGH_START = N_FULL
DEV16_HIGH_START = len(GMR_DEV_NAMES) + len(OGDEN_NAMES)       # 26: same idea in the dev16 layout


def full_names(n=N_FULL):
    """Names of a full vector of n columns (n > N_FULL: GMR terms of order > 3 at the end)."""
    return FULL_NAMES + [f"C{i}{j}" for i, j in gmr_high_terms(gmr_order_from_length(10 + n - N_FULL))][:max(0, n - N_FULL)]


def gmr_params_from_full(p):
    """GMR parameter vector [C10 .. E, C40, ...] of a full vector (any length)."""
    p = np.asarray(p)
    return np.concatenate([p[..., :10], p[..., HIGH_START:]], axis=-1) if p.shape[-1] > HIGH_START else p[..., :10]


def gmr_params_from_dev16(p):
    """GMR parameter vector [C10 .. E, C40, ...] of a dev16 vector (any length)."""
    p = np.asarray(p)
    return np.concatenate([p[..., :10], p[..., DEV16_HIGH_START:]], axis=-1) if p.shape[-1] > DEV16_HIGH_START else p[..., :10]


def ogden_from_dev16(p):
    """Ogden block (mu1, alpha1, ..., mu8, alpha8) of a dev16 vector."""
    return p[..., 10:DEV16_HIGH_START]


def is_dev_only(material_model):
    """'<library>_dev' (gmr_dev, ogden_dev, gmr_ogden_dev): deviatoric library only, no volumetric distillation."""
    return str(material_model or "").lower().endswith("_dev")


def library_from_material_model(material_model):
    name = str(material_model or "gmr").lower()
    name = name[:-4] if name.endswith("_dev") else name
    return {"ogden": "ogden", "gmr_ogden": "gmr_ogden"}.get(name, "gmr")


def distilled_gmr_order(distilled_dir):
    """GMR order of a distillation run (its saved --gmr_order; runs before the option: 3)."""
    for fname in ("dev_config.json", "config.json"):
        p = os.path.join(distilled_dir, fname)
        if os.path.exists(p):
            try:
                with open(p) as f:
                    return int(json.load(f).get("gmr_order", GMR_BASE_ORDER) or GMR_BASE_ORDER)
            except Exception:
                pass
    return GMR_BASE_ORDER


def distilled_library(distilled_dir):
    """Deviatoric library of a distillation run, from its saved configuration (default "gmr")."""
    for fname in ("dev_config.json", "config.json"):
        p = os.path.join(distilled_dir, fname)
        if os.path.exists(p):
            try:
                with open(p) as f:
                    return library_from_material_model(json.load(f).get("material_model"))
            except Exception:
                pass
    return "gmr"


def dev_names(library, order=GMR_BASE_ORDER):
    """Columns of the distilled dev samples of a library (GMR block of the given order first)."""
    return {"gmr": gmr_names(order), "ogden": OGDEN_NAMES, "gmr_ogden": gmr_names(order) + OGDEN_NAMES}[library]


def dev16_names(order=GMR_BASE_ORDER):
    """Columns of the dev16 layout (plots/plot_discovery_summary.py etc.)."""
    return GMR_DEV_NAMES + OGDEN_NAMES + [f"C{i}{j}" for i, j in gmr_high_terms(order)]


def full_param_vectors(dev, vol, library="gmr", order=None):
    """(S, n_dev), (S, 3) distilled samples -> full parameter vectors. "gmr" of order 3 keeps the former [dev, vol]
    layout (13 columns); Ogden libraries give the N_FULL-column layout; a GMR order > 3 appends its higher terms at
    HIGH_START. order: GMR order of the samples (None: from the number of columns, "gmr" only)."""
    dev, vol = np.atleast_2d(dev), np.atleast_2d(vol)
    if library == "ogden":
        ng = 0
    elif order is not None:
        ng = n_gmr(order)
    elif library == "gmr":
        ng = dev.shape[1]
    else:
        ng = dev.shape[1] - len(OGDEN_NAMES)
    if library == "gmr" and ng <= 10:
        return np.concatenate([dev, vol], axis=1)
    n_high = max(0, ng - 10)
    out = np.zeros((dev.shape[0], len(FULL_NAMES) + n_high))
    out[:, 10:13] = vol[:, :3]
    if ng:
        out[:, :min(10, ng)] = dev[:, :min(10, ng)]
        out[:, HIGH_START:HIGH_START + n_high] = dev[:, 10:ng]
    og = dev[:, ng:]
    out[:, OGDEN_START:OGDEN_START + og.shape[1]] = og
    return out


def true_full_vector(material_params):
    """Full vector of a true material from its config 'material_params' (GMR family or Ogden), or None."""
    mp = material_params or {}
    out = np.zeros(len(FULL_NAMES))
    if "dev_params" in mp:
        dp = list(mp["dev_params"])
        out[:min(10, len(dp))] = dp[:10]
        if len(dp) > 10:                         # GMR terms of order > 3
            out = np.concatenate([out, np.asarray(dp[10:], float)])
    if "vol_params" in mp:
        out[10:10 + len(mp["vol_params"])] = mp["vol_params"]
    if "aniso_params" in mp:
        out[13:13 + len(mp["aniso_params"])] = mp["aniso_params"]
    if "mu_params" in mp and "alpha_params" in mp:
        for k, (m, a) in enumerate(zip(mp["mu_params"], mp["alpha_params"])):
            out[OGDEN_START + 2 * k], out[OGDEN_START + 2 * k + 1] = m, a
    if not any(k in mp for k in ("dev_params", "mu_params")):
        return None
    return out


def dev16(dev, library="gmr", order=None):
    """Distilled dev samples -> (S, 26 [+ high]) [C10..E (10) | mu1, alpha1, ... (16) | C40, C31, ... (order > 3)] for
    energy evaluation (the name is historical: 16 columns with 3 Ogden terms). order: GMR order of the samples."""
    dev = np.atleast_2d(dev)
    if library == "ogden":
        ng = 0
    elif order is not None:
        ng = n_gmr(order)
    else:                                   # from the columns: GMR block first, the Ogden block (16) last
        ng = dev.shape[1] - (len(OGDEN_NAMES) if library == "gmr_ogden" else 0)
    n_high = max(0, ng - 10)
    out = np.zeros((dev.shape[0], DEV16_HIGH_START + n_high))
    if ng:
        out[:, :min(10, dev.shape[1])] = dev[:, :min(10, dev.shape[1])]
        out[:, DEV16_HIGH_START:] = dev[:, 10:ng]
    if library in ("ogden", "gmr_ogden"):
        og = dev[:, ng:]
        out[:, 10:10 + og.shape[1]] = og
    return out


def sort_ogden_terms(dev, library="gmr", mu_tol=1e-6):
    """
    Reorder the interchangeable Ogden terms of each sample for reporting: active terms (mu > mu_tol) first, by
    decreasing alpha; inactive terms (alpha meaningless) last. The energy is unchanged. Returns a copy.
    """
    dev = np.array(dev, dtype=float, copy=True)
    if library not in ("ogden", "gmr_ogden"):
        return dev
    start = 0 if library == "ogden" else dev.shape[1] - 2 * N_OGDEN_TERMS      # Ogden block last
    n_terms = (dev.shape[1] - start) // 2
    mu, al = dev[:, start:start + 2 * n_terms:2], dev[:, start + 1:start + 2 * n_terms:2]
    key = np.where(mu > mu_tol, -al, np.inf)                     # active by decreasing alpha, then inactive
    order = np.argsort(key, axis=1, kind="stable")
    rows = np.arange(dev.shape[0])[:, None]
    dev[:, start:start + 2 * n_terms:2], dev[:, start + 1:start + 2 * n_terms:2] = mu[rows, order], al[rows, order]
    return dev


def effective_moduli(full):
    """Small-strain moduli of full parameter vectors (S, 13 | N_FULL): mu = 2 (C10 + C01 + E/3) + sum mu_k (Ogden,
    convention 2 mu_k / alpha_k^2) and kappa = 2 D1 (psi_vol = D1 (J-1)^2 + ...). Same definition as the HSGP
    (mu = 2 (dpsi/dI1_bar + dpsi/dI2_bar), kappa = d2psi/dJ2 at F = I); fibre terms are not included."""
    full = np.atleast_2d(np.asarray(full, dtype=float))
    mu = 2.0 * (full[:, 0] + full[:, 1] + full[:, 9] / 3.0)
    if full.shape[1] > OGDEN_START:
        mu = mu + full[:, OGDEN_START:HIGH_START:2].sum(axis=1)     # GMR terms of order > 3: no small-strain part
    return mu, 2.0 * full[:, 10]


def true_effective_moduli(material):
    """(mu, kappa) of any material with psi(F (3, 3)) by differentiation at F = I: mu = d2psi/dgamma2 in isochoric
    simple shear, kappa = d2psi/dJ2 under pure dilatation F = J^(1/3) I."""
    import jax
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
    eye = jnp.eye(3, dtype=jnp.float64)
    shear = lambda g: material.psi(eye.at[0, 1].set(g))
    dil = lambda j: material.psi(j ** (1.0 / 3.0) * eye)
    mu = float(jax.grad(jax.grad(shear))(jnp.float64(0.0)))
    kappa = float(jax.grad(jax.grad(dil))(jnp.float64(1.0)))
    return mu, kappa
