"""
Candidate libraries of the distillation and the full parameter vector shared by the FEM validation and the plots.

Full vector layout (positions are fixed; shorter vectors mean the later blocks are zero):
    [C10, C01, C20, C11, C02, C30, C21, C12, C03, E | D1, D2, D3 | anisotropic (6) | mu1, alpha1, mu2, alpha2, mu3, alpha3]
     0 ............................................ 9  10 ..... 12  13 ......... 18  19 ......................... 24

Deviatoric libraries (the distilled dev samples): "gmr" polynomial (+ log) terms (10 columns), "ogden" three Ogden terms
(6 columns: mu1, alpha1, ...), "gmr_ogden" both (16 columns, polynomial first). The volumetric samples are D1..D3.
"""
import json
import os

import numpy as np

GMR_DEV_NAMES = ["C10", "C01", "C20", "C11", "C02", "C30", "C21", "C12", "C03", "E"]
VOL_NAMES = ["D1", "D2", "D3"]
ANISO_NAMES = ["C42", "C43", "C44", "C62", "C63", "C64"]
OGDEN_NAMES = ["mu1", "alpha1", "mu2", "alpha2", "mu3", "alpha3"]
FULL_NAMES = GMR_DEV_NAMES + VOL_NAMES + ANISO_NAMES + OGDEN_NAMES      # 25
OGDEN_START = 19


def is_dev_only(material_model):
    """'<library>_dev' (gmr_dev, ogden_dev, gmr_ogden_dev): deviatoric library only, no volumetric distillation."""
    return str(material_model or "").lower().endswith("_dev")


def library_from_material_model(material_model):
    name = str(material_model or "gmr").lower()
    name = name[:-4] if name.endswith("_dev") else name
    return {"ogden": "ogden", "gmr_ogden": "gmr_ogden"}.get(name, "gmr")


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


def dev_names(library):
    return {"gmr": GMR_DEV_NAMES, "ogden": OGDEN_NAMES, "gmr_ogden": GMR_DEV_NAMES + OGDEN_NAMES}[library]


def full_param_vectors(dev, vol, library="gmr"):
    """(S, n_dev), (S, 3) distilled samples -> full parameter vectors. "gmr" keeps the former [dev, vol] layout
    (13 columns); libraries with Ogden terms give the 25-column layout."""
    dev, vol = np.atleast_2d(dev), np.atleast_2d(vol)
    if library == "gmr":
        return np.concatenate([dev, vol], axis=1)
    out = np.zeros((dev.shape[0], len(FULL_NAMES)))
    out[:, 10:13] = vol[:, :3]
    if library == "ogden":
        out[:, OGDEN_START:OGDEN_START + dev.shape[1]] = dev
    else:
        out[:, :10] = dev[:, :10]
        out[:, OGDEN_START:OGDEN_START + dev.shape[1] - 10] = dev[:, 10:]
    return out


def true_full_vector(material_params):
    """Full vector of a true material from its config 'material_params' (GMR family or Ogden), or None."""
    mp = material_params or {}
    out = np.zeros(len(FULL_NAMES))
    if "dev_params" in mp:
        out[:len(mp["dev_params"])] = mp["dev_params"]
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


def dev16(dev, library="gmr"):
    """Distilled dev samples -> (S, 16) [C10..E (10) | mu1, alpha1, ..., alpha3 (6)] for energy evaluation."""
    dev = np.atleast_2d(dev)
    out = np.zeros((dev.shape[0], 16))
    if library == "ogden":
        out[:, 10:10 + dev.shape[1]] = dev
    elif library == "gmr_ogden":
        out[:, :dev.shape[1]] = dev
    else:
        out[:, :min(10, dev.shape[1])] = dev[:, :10]
    return out


def sort_ogden_terms(dev, library="gmr", mu_tol=1e-6):
    """
    Reorder the interchangeable Ogden terms of each sample for reporting: active terms (mu > mu_tol) first, by
    decreasing alpha; inactive terms (alpha meaningless) last. The energy is unchanged. Returns a copy.
    """
    dev = np.array(dev, dtype=float, copy=True)
    if library not in ("ogden", "gmr_ogden"):
        return dev
    start = 0 if library == "ogden" else 10
    n_terms = (dev.shape[1] - start) // 2
    mu, al = dev[:, start:start + 2 * n_terms:2], dev[:, start + 1:start + 2 * n_terms:2]
    key = np.where(mu > mu_tol, -al, np.inf)                     # active by decreasing alpha, then inactive
    order = np.argsort(key, axis=1, kind="stable")
    rows = np.arange(dev.shape[0])[:, None]
    dev[:, start:start + 2 * n_terms:2], dev[:, start + 1:start + 2 * n_terms:2] = mu[rows, order], al[rows, order]
    return dev


def effective_moduli(full):
    """Small-strain moduli of full parameter vectors (S, 13 | 25): mu = 2 (C10 + C01 + E/3) + sum mu_k (Ogden,
    convention 2 mu_k / alpha_k^2) and kappa = 2 D1 (psi_vol = D1 (J-1)^2 + ...). Same definition as the HSGP
    (mu = 2 (dpsi/dI1_bar + dpsi/dI2_bar), kappa = d2psi/dJ2 at F = I); fibre terms are not included."""
    full = np.atleast_2d(np.asarray(full, dtype=float))
    mu = 2.0 * (full[:, 0] + full[:, 1] + full[:, 9] / 3.0)
    if full.shape[1] > OGDEN_START:
        mu = mu + full[:, OGDEN_START::2].sum(axis=1)
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
