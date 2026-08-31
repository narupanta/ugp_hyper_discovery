# material_models.py
import os
from abc import ABC, abstractmethod
from typing import Dict, Type, Any, Optional, List, Tuple
import jax
import jax.numpy as jnp

from .utils import C_func, B_func, I1_func, I2_func, I3_func, J_func, load_model_config

# Registry
_material_registry: Dict[str, Type["BaseMaterialModel"]] = {}


def register_material(name: str):
    def decorator(cls):
        _material_registry[name.lower()] = cls
        return cls
    return decorator


def get_material(name: str, **kwargs) -> "BaseMaterialModel":
    name_clean = name.lower()
    if name_clean not in _material_registry:
        raise ValueError(f"Unknown material '{name}'. Available: {list(_material_registry.keys())}")
    return _material_registry[name_clean](**kwargs)


def get_material_from_config(config: Dict[str, Any], **kwargs) -> "BaseMaterialModel":
    """
    Instantiate a material model directly from a config dictionary (e.g., loaded from a YAML recipe or config.json).
    Strictly reads material parameters and raises ValueError if required parameters are missing.
    """
    model_name = config.get("material_model_name", None)
    if model_name is None:
        raise ValueError("Config dictionary must contain 'material_model_name'.")
    model_name = str(model_name).lower()
    
    mat_params = config.get("material_params", {})
    if not isinstance(mat_params, dict):
        mat_params = {}
    
    dev_params = mat_params.get("dev_params", config.get("dev_params", None))
    vol_params = mat_params.get("vol_params", config.get("vol_params", None))
    aniso_params = mat_params.get("aniso_params", config.get("aniso_params", None))
    angles = mat_params.get("angles", config.get("angles", None))
    
    mu_params = mat_params.get("mu_params", config.get("mu_params", None))
    alpha_params = mat_params.get("alpha_params", config.get("alpha_params", None))

    extra_kwargs = {**kwargs}
    if dev_params is not None: extra_kwargs["dev_params"] = dev_params
    if vol_params is not None: extra_kwargs["vol_params"] = vol_params
    if aniso_params is not None: extra_kwargs["aniso_params"] = aniso_params
    if angles is not None: extra_kwargs["angles"] = angles
    if mu_params is not None: extra_kwargs["mu_params"] = mu_params
    if alpha_params is not None: extra_kwargs["alpha_params"] = alpha_params

    return get_material(model_name, **extra_kwargs)


def get_material_from_dir(dir_path: str, **kwargs) -> "BaseMaterialModel":
    """
    Instantiate a material model by loading the configuration from a saved model directory.
    """
    cfg = load_model_config(dir_path)
    return get_material_from_config(cfg, **kwargs)


class BaseMaterialModel(ABC):
    """
    Base class: subclasses implement psi(F) (scalar per sample).
    P(F) is provided by this base class using JAX autograd:
        P_{iJ} = d psi / d F_{iJ}.
    This supports both single F (3,3) and batched F (...,3,3).
    """

    def __init__(self, jit_P: bool = True):
        self.jit_P = jit_P
        self._per_sample_grad = None
        self._batched_grad = None

    @abstractmethod
    def psi(self, F: jnp.ndarray) -> jnp.ndarray:
        raise NotImplementedError

    def psi_dev(self, F: jnp.ndarray) -> jnp.ndarray:
        return self.psi(F)

    def psi_vol(self, F: jnp.ndarray) -> jnp.ndarray:
        return jnp.zeros_like(self.psi(F))

    def psi_aniso(self, F: jnp.ndarray) -> jnp.ndarray:
        return jnp.zeros_like(self.psi(F))

    def _make_grad_fns(self) -> None:
        if self._per_sample_grad is not None:
            return

        def per_sample_phi(F_single: jnp.ndarray) -> jnp.ndarray:
            out = self.psi(F_single)
            return jnp.sum(out)

        per_sample_grad = jax.grad(per_sample_phi)
        if self.jit_P:
            per_sample_grad = jax.jit(per_sample_grad)

        def batched_grad(F_batched: jnp.ndarray) -> jnp.ndarray:
            orig_shape = F_batched.shape
            if F_batched.ndim == 2:
                return per_sample_grad(F_batched)
            leading = orig_shape[:-2]
            flat = F_batched.reshape((-1, orig_shape[-2], orig_shape[-1]))
            vmapped = jax.vmap(per_sample_grad)
            result_flat = vmapped(flat)
            return result_flat.reshape(*leading, orig_shape[-2], orig_shape[-1])

        if self.jit_P:
            batched_grad = jax.jit(batched_grad)

        self._per_sample_grad = per_sample_grad
        self._batched_grad = batched_grad

    def P(self, F: jnp.ndarray) -> jnp.ndarray:
        F = jnp.asarray(F)
        self._make_grad_fns()
        if F.ndim == 2:
            return self._per_sample_grad(F)
        else:
            return self._batched_grad(F)


# ------------------------------------------------------------------------------
# Unified Master Hyperelastic Model
# ------------------------------------------------------------------------------

@register_material("hyperelastic")
@register_material("custom_hyperelastic")
class HyperelasticModel(BaseMaterialModel):
    """
    Master candidate hyperelastic model combining:
    - Deviatoric terms: Polynomial expansion up to order 3 in (I1_bar - 3, I2_bar - 3) + log(I2_bar/3)
    - Volumetric terms: Polynomial expansion in (J - 1)^2, (J - 1)^4, (J - 1)^6
    - Anisotropic terms: Quadratic, Quartic, and Holzapfel terms for fiber families a0 and a1
    """

    def __init__(
        self,
        dev_params: Optional[List[float]] = None,
        vol_params: Optional[List[float]] = None,
        aniso_params: Optional[List[float]] = None,
        angles: Optional[List[float]] = None,
        a0: Optional[jnp.ndarray] = None,
        a1: Optional[jnp.ndarray] = None,
        cap_compression: bool = False,
        jit_P: bool = True,
    ):
        super().__init__(jit_P=jit_P)
        
        if dev_params is None:
            raise ValueError(
                "HyperelasticModel requires 'dev_params' to be explicitly provided (e.g. from recipe configuration). "
                "Default parameter fallbacks have been removed to prevent silent bugs."
            )
        if len(dev_params) != 10:
            raise ValueError(f"dev_params must have length exactly 10, but got length {len(dev_params)}: {dev_params}")
        self.dev_params = list(dev_params)

        if vol_params is None:
            raise ValueError(
                "HyperelasticModel requires 'vol_params' to be explicitly provided (e.g. from recipe configuration). "
                "Default parameter fallbacks have been removed to prevent silent bugs."
            )
        if len(vol_params) != 3:
            raise ValueError(f"vol_params must have length exactly 3, but got length {len(vol_params)}: {vol_params}")
        self.vol_params = list(vol_params)

        if aniso_params is not None:
            if len(aniso_params) != 6:
                raise ValueError(f"aniso_params must have length exactly 6, but got length {len(aniso_params)}: {aniso_params}")
            self.aniso_params = list(aniso_params)
        else:
            self.aniso_params = [0.0] * 6

        # If anisotropic parameters are active, enforce explicit fiber angles or vector definition
        has_aniso_active = any(abs(p) > 1e-12 for p in self.aniso_params)
        if has_aniso_active and (angles is None and a0 is None):
            raise ValueError(
                "Anisotropic parameters are active but no fiber 'angles' or 'a0' vectors were provided in configuration."
            )

        self.cap_compression = cap_compression

        # Resolve structural fiber vectors
        if a0 is not None:
            self.a0 = jnp.asarray(a0, dtype=jnp.float64)
            self.a1 = jnp.asarray(a1, dtype=jnp.float64) if a1 is not None else None
        elif angles is not None and len(angles) > 0:
            angles_rad = [float(jnp.radians(a)) if float(abs(a)) > 2.0 * float(jnp.pi) else float(a) for a in angles]
            self.a0 = jnp.array([jnp.cos(angles_rad[0]), jnp.sin(angles_rad[0]), 0.0], dtype=jnp.float64)
            if len(angles_rad) > 1:
                self.a1 = jnp.array([jnp.cos(angles_rad[1]), jnp.sin(angles_rad[1]), 0.0], dtype=jnp.float64)
            else:
                self.a1 = None
        else:
            self.a0 = None
            self.a1 = None

    def _eval_invariants(self, F: jnp.ndarray):
        if F.shape[-2:] == (2, 2):
            F_3d = jnp.array([[F[..., 0, 0], F[..., 0, 1], 0.0],
                              [F[..., 1, 0], F[..., 1, 1], 0.0],
                              [0.0,           0.0,           1.0]])
        else:
            F_3d = F
        C = C_func(F_3d)
        I1 = I1_func(C)
        I2 = I2_func(C)
        I3 = I3_func(C)
        I3_safe = jnp.clip(I3, 1.0e-8, 1.0e8)

        i1_dev = I3_safe ** (-1 / 3) * I1
        i2_dev = I3_safe ** (-2 / 3) * I2
        J = jnp.sqrt(I3_safe)
        C_bar = (I3_safe ** (-1 / 3)) * C
        return F_3d, C_bar, i1_dev, i2_dev, J

    def psi_dev(self, F: jnp.ndarray) -> jnp.ndarray:
        _, _, i1_dev, i2_dev, _ = self._eval_invariants(F)
        i1_dev_minus_3 = i1_dev - 3.0
        i2_dev_minus_3 = i2_dev - 3.0

        p = self.dev_params
        return (
            p[0] * i1_dev_minus_3 +
            p[1] * i2_dev_minus_3 +
            p[2] * i1_dev_minus_3**2 +
            p[3] * i1_dev_minus_3 * i2_dev_minus_3 +
            p[4] * i2_dev_minus_3**2 +
            p[5] * i1_dev_minus_3**3 +
            p[6] * (i1_dev_minus_3**2) * i2_dev_minus_3 +
            p[7] * i1_dev_minus_3 * (i2_dev_minus_3**2) +
            p[8] * i2_dev_minus_3**3 +
            p[9] * jnp.log(jnp.maximum(i2_dev / 3.0, 1e-8))
        )

    def psi_vol(self, F: jnp.ndarray) -> jnp.ndarray:
        _, _, _, _, J = self._eval_invariants(F)
        J_minus_1 = J - 1.0
        p = self.vol_params
        return (
            p[0] * J_minus_1**2 +
            p[1] * J_minus_1**4 +
            p[2] * J_minus_1**6
        )

    def psi_aniso(self, F: jnp.ndarray) -> jnp.ndarray:
        if self.a0 is None:
            return jnp.zeros(())
            
        _, C_bar, _, _, _ = self._eval_invariants(F)
        I4_bar = jnp.einsum('i,...ij,j->...', self.a0, C_bar, self.a0)
        I4_minus_1 = I4_bar - 1.0
        if self.cap_compression:
            I4_minus_1 = jnp.maximum(I4_minus_1, 0.0)

        I6_minus_1 = 0.0
        if self.a1 is not None:
            I6_bar = jnp.einsum('i,...ij,j->...', self.a1, C_bar, self.a1)
            I6_minus_1 = I6_bar - 1.0
            if self.cap_compression:
                I6_minus_1 = jnp.maximum(I6_minus_1, 0.0)

        p = self.aniso_params
        if len(p) == 6:
            # Polynomial anisotropic model
            aniso1 = p[0] * I4_minus_1**2 + p[1] * I4_minus_1**3 + p[2] * I4_minus_1**4
            aniso2 = p[3] * I6_minus_1**2 + p[4] * I6_minus_1**3 + p[5] * I6_minus_1**4 if self.a1 is not None else 0.0
            return aniso1 + aniso2
        else:
            # Extended / Exponential Holzapfel-type model
            c42 = p[0] if len(p) > 0 else 0.0
            c44 = p[1] if len(p) > 1 else 0.0
            k1  = p[2] if len(p) > 2 else 0.0
            k2  = p[3] if len(p) > 3 else 0.0
            c62 = p[4] if len(p) > 4 else 0.0
            c64 = p[5] if len(p) > 5 else 0.0
            k3  = p[6] if len(p) > 6 else 0.0
            k4  = p[7] if len(p) > 7 else 0.0

            aniso1 = c42 * I4_minus_1**2 + c44 * I4_minus_1**4
            if k2 != 0.0:
                aniso1 += (k1 / (2.0 * k2)) * (jnp.exp(k2 * I4_minus_1**2) - 1.0)

            aniso2 = 0.0
            if self.a1 is not None:
                aniso2 = c62 * I6_minus_1**2 + c64 * I6_minus_1**4
                if k4 != 0.0:
                    aniso2 += (k3 / (2.0 * k4)) * (jnp.exp(k4 * I6_minus_1**2) - 1.0)
            return aniso1 + aniso2

    def psi(self, F: jnp.ndarray) -> jnp.ndarray:
        return self.psi_dev(F) + self.psi_vol(F) + self.psi_aniso(F)


# ------------------------------------------------------------------------------
# Specific Material Models as Lightweight Parameter Configurations
# ------------------------------------------------------------------------------

@register_material("mooney-rivlin")
class MooneyRivlin(HyperelasticModel):
    def __init__(self, dev_params=None, vol_params=None, **kwargs):
        if dev_params is None or vol_params is None:
            raise ValueError("MooneyRivlin requires 'dev_params' and 'vol_params' from configuration.")
        super().__init__(dev_params=dev_params, vol_params=vol_params, **kwargs)


@register_material("neohookean")
class NeoHookean(HyperelasticModel):
    def __init__(self, dev_params=None, vol_params=None, c1=None, c2=None, **kwargs):
        if dev_params is None:
            if c1 is None:
                raise ValueError("NeoHookean requires 'dev_params' or 'c1' parameter from configuration.")
            dev_params = [c1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        if vol_params is None:
            if c2 is None:
                raise ValueError("NeoHookean requires 'vol_params' or 'c2' parameter from configuration.")
            vol_params = [c2, 0.0, 0.0]
        super().__init__(dev_params=dev_params, vol_params=vol_params, **kwargs)


@register_material("isihara")
class Isihara(HyperelasticModel):
    def __init__(self, dev_params=None, vol_params=None, c10=None, c01=None, c20=None, d1=None, **kwargs):
        if dev_params is None:
            if c10 is None or c01 is None or c20 is None:
                raise ValueError("Isihara requires 'dev_params' or (c10, c01, c20) from configuration.")
            dev_params = [c10, c01, c20, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        if vol_params is None:
            if d1 is None:
                raise ValueError("Isihara requires 'vol_params' or 'd1' from configuration.")
            vol_params = [d1, 0.0, 0.0]
        super().__init__(dev_params=dev_params, vol_params=vol_params, **kwargs)


@register_material("gmr")
@register_material("gmr_log")
@register_material("gmr_nolog")
@register_material("gmr_aniso")
class GeneralizedMooneyRivlin(HyperelasticModel):
    def __init__(self, dev_params=None, vol_params=None, aniso_params=None, angles=None, **kwargs):
        if dev_params is None or vol_params is None:
            raise ValueError("GeneralizedMooneyRivlin requires 'dev_params' and 'vol_params' from configuration.")
        super().__init__(dev_params=dev_params, vol_params=vol_params, aniso_params=aniso_params, angles=angles, **kwargs)


@register_material("gentthomas")
class GentThomas(HyperelasticModel):
    def __init__(self, dev_params=None, vol_params=None, c1=None, c2=None, **kwargs):
        if dev_params is None:
            if c1 is None:
                raise ValueError("GentThomas requires 'dev_params' or 'c1' from configuration.")
            dev_params = [c1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        if vol_params is None:
            if c2 is None:
                raise ValueError("GentThomas requires 'vol_params' or 'c2' from configuration.")
            vol_params = [c2, 0.0, 0.0]
        super().__init__(dev_params=dev_params, vol_params=vol_params, **kwargs)


@register_material("nh4")
@register_material("neohookean4")
class NeoHookean4(HyperelasticModel):
    def __init__(self, dev_params=None, vol_params=None, c10=None, d2=None, **kwargs):
        if dev_params is None:
            if c10 is None:
                raise ValueError("NeoHookean4 requires 'dev_params' or 'c10' from configuration.")
            dev_params = [c10, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        if vol_params is None:
            if d2 is None:
                raise ValueError("NeoHookean4 requires 'vol_params' or 'd2' from configuration.")
            vol_params = [0.0, d2, 0.0]
        super().__init__(dev_params=dev_params, vol_params=vol_params, **kwargs)


@register_material("nh2")
@register_material("neohookean2")
@register_material("nh")
class NeoHookeanGMR(HyperelasticModel):
    def __init__(self, dev_params=None, vol_params=None, c10=None, d1=None, **kwargs):
        if dev_params is None:
            if c10 is None:
                raise ValueError("NeoHookeanGMR requires 'dev_params' or 'c10' from configuration.")
            dev_params = [c10, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        if vol_params is None:
            if d1 is None:
                raise ValueError("NeoHookeanGMR requires 'vol_params' or 'd1' from configuration.")
            vol_params = [d1, 0.0, 0.0]
        super().__init__(dev_params=dev_params, vol_params=vol_params, **kwargs)


@register_material("c20_d10_d05")
@register_material("c20d10d05")
class CustomGT(HyperelasticModel):
    def __init__(self, dev_params=None, vol_params=None, c10=None, d1=None, d2=None, **kwargs):
        if dev_params is None:
            if c10 is None:
                raise ValueError("CustomGT requires 'dev_params' or 'c10' from configuration.")
            dev_params = [c10, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        if vol_params is None:
            if d1 is None or d2 is None:
                raise ValueError("CustomGT requires 'vol_params' or (d1, d2) from configuration.")
            vol_params = [d1, d2, 0.0]
        super().__init__(dev_params=dev_params, vol_params=vol_params, **kwargs)


@register_material("aniso30")
class Aniso30(HyperelasticModel):
    def __init__(self, dev_params=None, vol_params=None, aniso_params=None, angles=None, **kwargs):
        if dev_params is None or vol_params is None or aniso_params is None or angles is None:
            raise ValueError("Aniso30 requires 'dev_params', 'vol_params', 'aniso_params', and 'angles' from configuration.")
        super().__init__(dev_params=dev_params, vol_params=vol_params, aniso_params=aniso_params, angles=angles, **kwargs)


@register_material("aniso45")
class Aniso45(HyperelasticModel):
    def __init__(self, dev_params=None, vol_params=None, aniso_params=None, angles=None, **kwargs):
        if dev_params is None or vol_params is None or aniso_params is None or angles is None:
            raise ValueError("Aniso45 requires 'dev_params', 'vol_params', 'aniso_params', and 'angles' from configuration.")
        super().__init__(dev_params=dev_params, vol_params=vol_params, aniso_params=aniso_params, angles=angles, **kwargs)


@register_material("ortho45")
class Ortho45(HyperelasticModel):
    def __init__(self, dev_params=None, vol_params=None, aniso_params=None, angles=None, **kwargs):
        if dev_params is None or vol_params is None or aniso_params is None or angles is None:
            raise ValueError("Ortho45 requires 'dev_params', 'vol_params', 'aniso_params', and 'angles' from configuration.")
        super().__init__(dev_params=dev_params, vol_params=vol_params, aniso_params=aniso_params, angles=angles, **kwargs)


@register_material("ortho090")
class Ortho090(HyperelasticModel):
    def __init__(self, dev_params=None, vol_params=None, aniso_params=None, angles=None, **kwargs):
        if dev_params is None or vol_params is None or aniso_params is None or angles is None:
            raise ValueError("Ortho090 requires 'dev_params', 'vol_params', 'aniso_params', and 'angles' from configuration.")
        super().__init__(dev_params=dev_params, vol_params=vol_params, aniso_params=aniso_params, angles=angles, **kwargs)


@register_material("ortho900")
class Ortho900(HyperelasticModel):
    def __init__(self, dev_params=None, vol_params=None, aniso_params=None, angles=None, **kwargs):
        if dev_params is None or vol_params is None or aniso_params is None or angles is None:
            raise ValueError("Ortho900 requires 'dev_params', 'vol_params', 'aniso_params', and 'angles' from configuration.")
        super().__init__(dev_params=dev_params, vol_params=vol_params, aniso_params=aniso_params, angles=angles, **kwargs)


@register_material("symnonortho60")
class SymNonOrtho60(HyperelasticModel):
    def __init__(self, dev_params=None, vol_params=None, aniso_params=None, angles=None, **kwargs):
        if dev_params is None or vol_params is None or aniso_params is None or angles is None:
            raise ValueError("SymNonOrtho60 requires 'dev_params', 'vol_params', 'aniso_params', and 'angles' from configuration.")
        super().__init__(dev_params=dev_params, vol_params=vol_params, aniso_params=aniso_params, angles=angles, **kwargs)


@register_material("ogden")
class Ogden(BaseMaterialModel):
    def __init__(self, mu_params=None, alpha_params=None, vol_params=None, jit_P: bool = True):
        super().__init__(jit_P=jit_P)
        if mu_params is None or alpha_params is None or vol_params is None:
            raise ValueError("Ogden model requires 'mu_params', 'alpha_params', and 'vol_params' from configuration.")
        self.mu_params = mu_params
        self.alpha_params = alpha_params
        self.vol_params = vol_params

    def psi(self, F: jnp.ndarray) -> jnp.ndarray:
        if F.shape[-2:] == (2, 2):
            F = jnp.array([[F[..., 0, 0], F[..., 0, 1], 0.0],
                           [F[..., 1, 0], F[..., 1, 1], 0.0],
                           [0.0,           0.0,           1.0]])
        C = C_func(F)
        I3 = I3_func(C)
        I3_safe = jnp.clip(I3, 1.0e-8, 1.0e8)

        eigenvalues = jnp.linalg.eigvalsh(C)
        eigenvalues = jnp.clip(eigenvalues, 1.0e-8, 1.0e8)
        lam = jnp.sqrt(eigenvalues)
        lam_bar = lam * I3_safe ** (-1 / 6)

        dev_terms = 0.0
        for i in range(len(self.mu_params)):
            val = (lam_bar[..., 0]**self.alpha_params[i] + lam_bar[..., 1]**self.alpha_params[i] + lam_bar[..., 2]**self.alpha_params[i] - 3.0)
            alpha_safe = jnp.where(jnp.abs(self.alpha_params[i]) < 1e-6, 1e-6 * jnp.sign(self.alpha_params[i] + 1e-9), self.alpha_params[i])
            dev_terms += (self.mu_params[i] / alpha_safe) * val

        J = jnp.sqrt(I3_safe)
        J_minus_1 = J - 1.0

        vol_terms = (
            self.vol_params[0] * J_minus_1**2 +
            (self.vol_params[1] * J_minus_1**4 if len(self.vol_params) > 1 else 0.0) +
            (self.vol_params[2] * J_minus_1**6 if len(self.vol_params) > 2 else 0.0)
        )
        return dev_terms + vol_terms


@register_material("holzapfel")
class AnisotropicHolzapfel(HyperelasticModel):
    """
    Anisotropic Holzapfel-Gasser-Ogden material model.
    """

    def __init__(self, dev_params=None, vol_params=None, aniso_params=None, angles=None, kappa=0.0, **kwargs):
        if dev_params is None or vol_params is None or aniso_params is None or angles is None:
            raise ValueError("AnisotropicHolzapfel requires 'dev_params', 'vol_params', 'aniso_params', and 'angles' from configuration.")
        super().__init__(
            dev_params=dev_params,
            vol_params=vol_params,
            aniso_params=aniso_params,
            angles=angles,
            **kwargs
        )
        self.k1 = self.aniso_params[2] if len(self.aniso_params) > 2 else 0.0
        self.k2 = self.aniso_params[3] if len(self.aniso_params) > 3 else 0.0
        self.kappa = kappa

    def psi_aniso(self, F: jnp.ndarray) -> jnp.ndarray:
        if self.a0 is None:
            return jnp.zeros(())
        _, C_bar, i1_dev, _, _ = self._eval_invariants(F)
        I4_bar = jnp.einsum('i,...ij,j->...', self.a0, C_bar, self.a0)
        E_bar = self.kappa * (i1_dev - 3.0) + (1.0 - 3.0 * self.kappa) * (I4_bar - 1.0)
        E_active = jnp.maximum(E_bar, 0.0)
        if self.k2 == 0.0:
            return 0.5 * self.k1 * (E_active**2)
        return (self.k1 / (2.0 * self.k2)) * (jnp.exp(self.k2 * (E_active**2)) - 1.0)
