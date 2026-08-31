# material_models.py
from abc import ABC, abstractmethod
from typing import Dict, Type, Any, Optional, List, Tuple
import jax
import jax.numpy as jnp

from .utils import C_func, B_func, I1_func, I2_func, I3_func, J_func

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
    Instantiate a material model directly from a config dictionary (e.g., loaded from a YAML recipe).
    Supports either pre-registered model names or custom HyperelasticModel parameter blocks.
    """
    model_name = config.get("material_model_name", "neohookean").lower()
    mat_params = config.get("material_params", {})
    if not isinstance(mat_params, dict):
        mat_params = {}
    
    dev_params = mat_params.get("dev_params", config.get("dev_params", None))
    vol_params = mat_params.get("vol_params", config.get("vol_params", None))
    aniso_params = mat_params.get("aniso_params", config.get("aniso_params", None))
    angles = mat_params.get("angles", config.get("angles", None))
    
    if model_name in ["hyperelastic", "custom_hyperelastic"]:
        return HyperelasticModel(
            dev_params=dev_params,
            vol_params=vol_params,
            aniso_params=aniso_params,
            angles=angles,
            **kwargs
        )
    else:
        extra_kwargs = {**kwargs}
        if dev_params is not None: extra_kwargs["dev_params"] = dev_params
        if vol_params is not None: extra_kwargs["vol_params"] = vol_params
        if aniso_params is not None: extra_kwargs["aniso_params"] = aniso_params
        if angles is not None: extra_kwargs["angles"] = angles
        return get_material(model_name, **extra_kwargs)


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
        
        # If parameters are None, default to 0s as specified
        self.dev_params = list(dev_params) if dev_params is not None else [0.0] * 10
        if len(self.dev_params) < 10:
            self.dev_params += [0.0] * (10 - len(self.dev_params))

        self.vol_params = list(vol_params) if vol_params is not None else [0.0] * 3
        if len(self.vol_params) < 3:
            self.vol_params += [0.0] * (3 - len(self.vol_params))

        if aniso_params is not None:
            self.aniso_params = list(aniso_params)
        else:
            self.aniso_params = [0.0] * 6

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
        C_bar = (I3_safe ** (-1 / 3))[..., None, None] * C
        return F_3d, C_bar, i1_dev, i2_dev, J

    def psi_dev(self, F: jnp.ndarray) -> jnp.ndarray:
        _, _, i1_dev, i2_dev, _ = self._eval_invariants(F)
        X = i1_dev - 3.0
        Y = i2_dev - 3.0
        log2 = jnp.log(jnp.clip(i2_dev / 3.0, 1.0e-8, 1.0e8))

        dev_terms = (
            self.dev_params[0] * X +
            self.dev_params[1] * Y +
            self.dev_params[2] * X**2 +
            self.dev_params[3] * X * Y +
            self.dev_params[4] * Y**2 +
            self.dev_params[5] * X**3 +
            self.dev_params[6] * (X**2) * Y +
            self.dev_params[7] * X * (Y**2) +
            self.dev_params[8] * Y**3 +
            self.dev_params[9] * log2
        )
        return dev_terms

    def psi_vol(self, F: jnp.ndarray) -> jnp.ndarray:
        _, _, _, _, J = self._eval_invariants(F)
        J_minus_1 = J - 1.0
        vol_terms = (
            self.vol_params[0] * J_minus_1**2 +
            self.vol_params[1] * J_minus_1**4 +
            self.vol_params[2] * J_minus_1**6
        )
        return vol_terms

    def psi_aniso(self, F: jnp.ndarray) -> jnp.ndarray:
        if self.a0 is None or self.aniso_params is None or len(self.aniso_params) == 0:
            return jnp.zeros(())
        _, C_bar, _, _, _ = self._eval_invariants(F)

        I4_bar_1 = jnp.einsum('i,...ij,j->...', self.a0, C_bar, self.a0)
        I4_m1 = I4_bar_1 - 1.0

        if len(self.aniso_params) == 6:
            C42, C43, C44, C62, C63, C64 = self.aniso_params
            aniso1 = C42 * I4_m1**2 + C43 * I4_m1**3 + C44 * I4_m1**4
            aniso2 = 0.0
            if self.a1 is not None:
                I4_bar_2 = jnp.einsum('i,...ij,j->...', self.a1, C_bar, self.a1)
                I6_m1 = I4_bar_2 - 1.0
                aniso2 = C62 * I6_m1**2 + C63 * I6_m1**3 + C64 * I6_m1**4
            return aniso1 + aniso2
        else:
            ta = list(self.aniso_params) + [0.0] * (8 - len(self.aniso_params))
            C42, C44, k1, k2 = ta[0], ta[1], ta[2], ta[3]
            exp_arg1 = jnp.clip(k2 * I4_m1**2, -30.0, 30.0)
            aniso1 = (C42 * I4_m1**2 + C44 * I4_m1**4 + k1 * (jnp.exp(exp_arg1) - 1.0))

            aniso2 = 0.0
            if self.a1 is not None:
                I4_bar_2 = jnp.einsum('i,...ij,j->...', self.a1, C_bar, self.a1)
                I6_m1 = I4_bar_2 - 1.0
                C62, C64, k3, k4 = ta[4], ta[5], ta[6], ta[7]
                exp_arg2 = jnp.clip(k4 * I6_m1**2, -30.0, 30.0)
                aniso2 = (C62 * I6_m1**2 + C64 * I6_m1**4 + k3 * (jnp.exp(exp_arg2) - 1.0))

            return aniso1 + aniso2

    def psi(self, F: jnp.ndarray) -> jnp.ndarray:
        return self.psi_dev(F) + self.psi_vol(F) + self.psi_aniso(F)


# ------------------------------------------------------------------------------
# Specific Material Models as Lightweight Parameter Configurations
# ------------------------------------------------------------------------------

@register_material("mooney-rivlin")
class MooneyRivlin(HyperelasticModel):
    def __init__(self, dev_params=None, vol_params=None, **kwargs):
        if dev_params is None:
            dev_params = [0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        if vol_params is None:
            vol_params = [1.5, 0.0, 0.0]
        super().__init__(dev_params=dev_params, vol_params=vol_params, **kwargs)


@register_material("neohookean")
class NeoHookean(HyperelasticModel):
    def __init__(self, c1=1.0, c2=1.5, **kwargs):
        super().__init__(dev_params=[c1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], vol_params=[c2, 0.0, 0.0], **kwargs)
        self.c1 = c1
        self.c2 = c2


@register_material("isihara")
class Isihara(HyperelasticModel):
    def __init__(self, c10=0.5, c01=1.0, c20=1.0, d1=1.5, **kwargs):
        super().__init__(dev_params=[c10, c01, c20, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], vol_params=[d1, 0.0, 0.0], **kwargs)


@register_material("gmr")
@register_material("gmr_log")
@register_material("gmr_nolog")
class GeneralizedMooneyRivlin(HyperelasticModel):
    def __init__(self, dev_params=None, vol_params=None, **kwargs):
        if dev_params is None:
            dev_params = [0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        if vol_params is None:
            vol_params = [1.5, 0.0, 0.0]
        super().__init__(dev_params=dev_params, vol_params=vol_params, **kwargs)


@register_material("gentthomas")
class GentThomas(HyperelasticModel):
    def __init__(self, c1=0.5, c2=1.5, **kwargs):
        super().__init__(dev_params=[c1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], vol_params=[c2, 0.0, 0.0], **kwargs)


@register_material("nh4")
@register_material("neohookean4")
class NeoHookean4(HyperelasticModel):
    def __init__(self, c10=0.5, d2=1.5, **kwargs):
        super().__init__(dev_params=[c10, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], vol_params=[0.0, d2, 0.0], **kwargs)


@register_material("nh2")
@register_material("neohookean2")
@register_material("nh")
class NeoHookeanGMR(HyperelasticModel):
    def __init__(self, c10=0.5, d1=1.5, **kwargs):
        super().__init__(dev_params=[c10, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], vol_params=[d1, 0.0, 0.0], **kwargs)
        self.c10 = c10
        self.d1 = d1


@register_material("c20_d10_d05")
@register_material("c20d10d05")
class CustomGT(HyperelasticModel):
    def __init__(self, c10=2.0, d1=1.0, d2=0.5, **kwargs):
        super().__init__(dev_params=[c10, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], vol_params=[d1, d2, 0.0], **kwargs)


@register_material("aniso30")
class Aniso30(HyperelasticModel):
    def __init__(self, c10=0.5, d1=1.0, c42=0.7, theta=jnp.pi/6.0, angles=None, dev_params=None, vol_params=None, aniso_params=None, **kwargs):
        if angles is None:
            angles = [float(jnp.degrees(theta))]
        if dev_params is None:
            dev_params = [c10, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        if vol_params is None:
            vol_params = [d1, 0.0, 0.0]
        if aniso_params is None:
            aniso_params = [c42, 0.0, 0.0, 0.0, 0.0, 0.0]
        super().__init__(
            dev_params=dev_params,
            vol_params=vol_params,
            aniso_params=aniso_params,
            angles=angles,
            **kwargs
        )
        self.c10 = self.dev_params[0]
        self.d1 = self.vol_params[0]
        self.c42 = self.aniso_params[0]


@register_material("ortho45")
class Ortho45(HyperelasticModel):
    def __init__(self, c10=0.5, d1=1.0, c42=0.7, c62=0.9, theta1=jnp.pi/4.0, theta2=-jnp.pi/4.0, angles=None, dev_params=None, vol_params=None, aniso_params=None, **kwargs):
        if angles is None:
            angles = [float(jnp.degrees(theta1)), float(jnp.degrees(theta2))]
        if dev_params is None:
            dev_params = [c10, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        if vol_params is None:
            vol_params = [d1, 0.0, 0.0]
        if aniso_params is None:
            aniso_params = [c42, 0.0, 0.0, c62, 0.0, 0.0]
        super().__init__(
            dev_params=dev_params,
            vol_params=vol_params,
            aniso_params=aniso_params,
            angles=angles,
            **kwargs
        )
        self.c10 = self.dev_params[0]
        self.d1 = self.vol_params[0]
        self.c42 = self.aniso_params[0]
        self.c62 = self.aniso_params[3] if len(self.aniso_params) > 3 else self.aniso_params[-1]


@register_material("ortho090")
class Ortho090(HyperelasticModel):
    def __init__(self, c10=0.5, d1=1.0, c42=0.7, c62=0.7, angles=None, dev_params=None, vol_params=None, aniso_params=None, **kwargs):
        if angles is None:
            angles = [0.0, 90.0]
        if dev_params is None:
            dev_params = [c10, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        if vol_params is None:
            vol_params = [d1, 0.0, 0.0]
        if aniso_params is None:
            aniso_params = [c42, 0.0, 0.0, c62, 0.0, 0.0]
        super().__init__(
            dev_params=dev_params,
            vol_params=vol_params,
            aniso_params=aniso_params,
            angles=angles,
            **kwargs
        )
        self.c10 = self.dev_params[0]
        self.d1 = self.vol_params[0]
        self.c42 = self.aniso_params[0]
        self.c62 = self.aniso_params[3] if len(self.aniso_params) > 3 else self.aniso_params[-1]


@register_material("ortho900")
class Ortho900(HyperelasticModel):
    def __init__(self, c10=0.5, d1=1.0, c42=0.7, c62=0.7, angles=None, dev_params=None, vol_params=None, aniso_params=None, **kwargs):
        if angles is None:
            angles = [90.0, 0.0]
        if dev_params is None:
            dev_params = [c10, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        if vol_params is None:
            vol_params = [d1, 0.0, 0.0]
        if aniso_params is None:
            aniso_params = [c42, 0.0, 0.0, c62, 0.0, 0.0]
        super().__init__(
            dev_params=dev_params,
            vol_params=vol_params,
            aniso_params=aniso_params,
            angles=angles,
            **kwargs
        )
        self.c10 = self.dev_params[0]
        self.d1 = self.vol_params[0]
        self.c42 = self.aniso_params[0]
        self.c62 = self.aniso_params[3] if len(self.aniso_params) > 3 else self.aniso_params[-1]


@register_material("symnonortho60")
class SymNonOrtho60(HyperelasticModel):
    def __init__(self, c10=0.5, d1=1.0, c42=0.7, c62=0.9, theta1=jnp.pi/3.0, theta2=-jnp.pi/3.0, angles=None, dev_params=None, vol_params=None, aniso_params=None, **kwargs):
        if angles is None:
            angles = [float(jnp.degrees(theta1)), float(jnp.degrees(theta2))]
        if dev_params is None:
            dev_params = [c10, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        if vol_params is None:
            vol_params = [d1, 0.0, 0.0]
        if aniso_params is None:
            aniso_params = [c42, 0.0, 0.0, c62, 0.0, 0.0]
        super().__init__(
            dev_params=dev_params,
            vol_params=vol_params,
            aniso_params=aniso_params,
            angles=angles,
            **kwargs
        )
        self.c10 = self.dev_params[0]
        self.d1 = self.vol_params[0]
        self.c42 = self.aniso_params[0]
        self.c62 = self.aniso_params[3] if len(self.aniso_params) > 3 else self.aniso_params[-1]


@register_material("ogden")
class Ogden(BaseMaterialModel):
    def __init__(self, mu_params=None, alpha_params=None, vol_params=None, jit_P: bool = True):
        super().__init__(jit_P=jit_P)
        self.mu_params = mu_params if mu_params is not None else [1.0, 0.0, 0.0]
        self.alpha_params = alpha_params if alpha_params is not None else [2.0, 0.0, 0.0]
        self.vol_params = vol_params if vol_params is not None else [1.5, 0.0, 0.0]

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
    Anisotropic Holzapfel-Gasser-Ogden material model with 1 fiber orientation.
    Uses Macauley bracket to ensure fibers do not support compression.
    """

    def __init__(self, c10=0.5, d1=1.5, k1=0.8, k2=0.9, kappa=0.0, theta=jnp.pi/6.0, angles=None, **kwargs):
        if angles is None:
            angles = [float(jnp.degrees(theta))]
        super().__init__(
            dev_params=[c10, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            vol_params=[d1, 0.0, 0.0],
            aniso_params=[0.0, 0.0, k1, k2, 0.0, 0.0, 0.0, 0.0],
            angles=angles,
            **kwargs
        )
        self.c10 = c10
        self.d1 = d1
        self.k1 = k1
        self.k2 = k2
        self.kappa = kappa

    def psi_aniso(self, F: jnp.ndarray) -> jnp.ndarray:
        if self.a0 is None:
            return jnp.zeros(())
        _, C_bar, i1_dev, _, _ = self._eval_invariants(F)
        I4_bar = jnp.einsum('i,...ij,j->...', self.a0, C_bar, self.a0)
        E_bar = self.kappa * (i1_dev - 3.0) + (1.0 - 3.0 * self.kappa) * (I4_bar - 1.0)
        E_active = jnp.maximum(E_bar, 0.0)
        return (self.k1 / (2.0 * self.k2)) * (jnp.exp(self.k2 * (E_active**2)) - 1.0)
