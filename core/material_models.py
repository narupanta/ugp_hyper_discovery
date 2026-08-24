# material_models_autodiff.py
from abc import ABC, abstractmethod
from typing import Dict, Type, Any
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
    name = name.lower()
    if name not in _material_registry:
        raise ValueError(f"Unknown material '{name}'. Available: {list(_material_registry.keys())}")
    return _material_registry[name](**kwargs)


class BaseMaterialModel(ABC):
    """
    Base class: subclasses implement phi(F) (scalar per sample).
    P(F) is provided by this base class using JAX autograd:
        P_{iJ} = d phi / d F_{iJ}.
    This supports both single F (3,3) and batched F (...,3,3).
    """

    def __init__(self, jit_P: bool = True):
        """
        jit_P: if True, the computed P function will be jitted/vmap'ed for speed.
        """
        self.jit_P = jit_P
        # cached compiled functions (lazy)
        self._per_sample_grad = None
        self._batched_grad = None

    @abstractmethod
    def psi(self, F: jnp.ndarray) -> jnp.ndarray:
        raise NotImplementedError

    def psi_dev(self, F: jnp.ndarray) -> jnp.ndarray:
        return self.psi(F) # Default fallback

    def psi_vol(self, F: jnp.ndarray) -> jnp.ndarray:
        return jnp.zeros_like(self.psi(F)) # Default fallback

    def _make_grad_fns(self) -> None:
        """
        Create and cache grad functions for per-sample and batched inputs.
        We wrap phi so that the function passed to grad returns a scalar for a single 3x3 F.
        """
        if self._per_sample_grad is not None:
            return  # already created

        # per-sample scalar phi function: ensures a scalar for a single (3,3) input
        def per_sample_phi(F_single: jnp.ndarray) -> jnp.ndarray:
            # ensure phi returns a scalar; if phi returns non-scalar, sum it
            out = self.psi(F_single)
            return jnp.sum(out)

        # gradient of scalar w.r.t. F (returns same shape as F_single)
        per_sample_grad = jax.grad(per_sample_phi)

        # optionally jit the single-sample grad
        if self.jit_P:
            per_sample_grad = jax.jit(per_sample_grad)

        # batched version using vmap across leading flattened batch axis
        def batched_grad(F_batched: jnp.ndarray) -> jnp.ndarray:
            # F_batched shape (..., 3, 3)
            # flatten leading dims to (N,3,3), vmap per_sample_grad, then reshape back
            orig_shape = F_batched.shape
            if F_batched.ndim == 2:
                # single sample passed accidentally; delegate to per-sample
                return per_sample_grad(F_batched)
            leading = orig_shape[:-2]
            flat = F_batched.reshape((-1, orig_shape[-2], orig_shape[-1]))  # (N,3,3)
            # vmap the (possibly jitted) per-sample grad
            vmapped = jax.vmap(per_sample_grad)
            result_flat = vmapped(flat)  # (N,3,3)
            return result_flat.reshape(*leading, orig_shape[-2], orig_shape[-1])

        # optionally jit batched_grad for performance
        if self.jit_P:
            batched_grad = jax.jit(batched_grad)

        # cache
        self._per_sample_grad = per_sample_grad
        self._batched_grad = batched_grad

    def P(self, F: jnp.ndarray) -> jnp.ndarray:
        """
        Compute 1st Piola-Kirchhoff stress P = d(phi)/dF.
        Accepts single F (3,3) or batched F (...,3,3). Returns same leading shape with last two dims (3,3).
        """
        F = jnp.asarray(F)
        self._make_grad_fns()
        # pick correct function based on input rank
        if F.ndim == 2:  # (3,3) single sample
            return self._per_sample_grad(F)
        else:
            return self._batched_grad(F)


# ------------------------------
# Mooney–Rivlin and Neo-Hookean that only implement phi()
# ------------------------------

@register_material("mooney-rivlin")
class MooneyRivlin(BaseMaterialModel):
    def __init__(self, dev_params, vol_params, jit_P: bool = True):
        super().__init__(jit_P=jit_P)
        self.dev_params = dev_params
        self.vol_params = vol_params

    def psi(self, F: jnp.ndarray) -> jnp.ndarray:
        if F.shape[-2:] == (2, 2):
            F = jnp.array([[F[0, 0], F[0, 1], 0.], 
                        [F[1, 0], F[1, 1], 0.],
                        [0.,      0.,     1. ]])
        c = C_func(F)
        I1 = I1_func(c)
        I2 = I2_func(c)
        I3 = I3_func(c)
        I3_safe = jnp.clip(I3, 1.0e-8, 1.0e8)
        i1_dev = I3_safe**(-1/3) * I1
        i2_dev = I3_safe**(-2/3) * I2

        X = i1_dev - 3.0
        Y = i2_dev - 3.0
        
        # --- Deviatoric Terms (W) ---
        # Assuming dev_params = [c01, c02, c10, c11, c12, c20, c21, c22]
        # Using the standard N=2 Polynomial Model terms (C10, C01, C20, C11, C02)
        dev_terms = (
            # C10 * X
            self.dev_params[2] * X + 
            # C01 * Y
            self.dev_params[0] * Y + 
            # C20 * X**2
            self.dev_params[5] * X**2 + 
            # C11 * X * Y
            self.dev_params[3] * X * Y + 
            # C02 * Y**2
            self.dev_params[1] * Y**2 +

            self.dev_params[4] * X*Y**2 + 

            self.dev_params[6] * X**2 * Y + 

            self.dev_params[7] * X**2 * Y ** 2

            # Add C12, C21, C22 terms here if required by your specific model definition
        )
        
        # --- Volumetric Terms (U) ---
        # Assuming vol_params = [d0, d1] are D2 and D1 parameters (inverse bulk moduli)
        J = jnp.sqrt(I3_safe)
        J_minus_1 = J - 1.0

        # Assuming the volumetric function U(J) = (1/D1)(J-1)^2 + (1/D2)(J-1)^4
        # with D1=d1 and D2=d0 (or vice versa, depending on convention)
        
        # D1 is typically the lower order term (quadratic, hence d1)
        # D2 is typically the higher order term (quartic, hence d0)
        vol_terms = (
            # (1/D1) * (J - 1)**2
            (self.vol_params[0]) * J_minus_1**2 + 
            # (1/D2) * (J - 1)**4
            (self.vol_params[1]) * J_minus_1**4
        )
        
        return dev_terms + vol_terms


@register_material("neohookean")
class NeoHookean(BaseMaterialModel):
    def __init__(self, c1=1.0, c2=1.5, jit_P: bool = True):
        super().__init__(jit_P=jit_P)
        self.c1 = c1
        self.c2 = c2

    def psi(self, F: jnp.ndarray) -> jnp.ndarray:
        C = C_func(F)
        I1 = I1_func(C)
        I3 = I3_func(C)
        I3_safe = jnp.clip(I3, 1.0e-8, 1.0e8)
        term1 = self.c1 * (I3_safe**(-1/3) * I1 - 3)
        term2 = self.c2 * (jnp.sqrt(I3_safe) - 1)**2
        return term1 + term2


@register_material("isihara")
class Isihara(BaseMaterialModel):
    def __init__(self, c10=0.5, c01=1.0, c20=1.0, d1=1.5, jit_P: bool = True):
        super().__init__(jit_P=jit_P)
        self.c10 = c10
        self.c01 = c01
        self.c20 = c20
        self.d1 = d1

    def psi(self, F: jnp.ndarray) -> jnp.ndarray:
        return self.psi_dev(F) + self.psi_vol(F)

    def psi_dev(self, F: jnp.ndarray) -> jnp.ndarray:
        C = C_func(F)
        I1 = I1_func(C)
        I2 = I2_func(C)
        I3 = I3_func(C)
        I3_safe = jnp.clip(I3, 1.0e-8, 1.0e8)
        
        term1 = self.c10 * (I3_safe**(-1/3) * I1 - 3)
        term2 = self.c01 * (I3_safe**(-2/3) * I2 - 3)
        term3 = self.c20 * (I3_safe**(-1/3) * I1 - 3)**2
        return term1 + term2 + term3

    def psi_vol(self, F: jnp.ndarray) -> jnp.ndarray:
        C = C_func(F)
        I3 = I3_func(C)
        I3_safe = jnp.clip(I3, 1.0e-8, 1.0e8)
        return self.d1 * (jnp.sqrt(I3_safe) - 1)**2

@register_material("gmr")
@register_material("gmr_log")
@register_material("gmr_nolog")
class GeneralizedMooneyRivlin(BaseMaterialModel):
    def __init__(self, dev_params=None, vol_params=None, jit_P: bool = True):
        super().__init__(jit_P=jit_P)
        # dev_params: [C10, C01, C20, C11, C02, C30, C21, C12, C03, CL1, CL2]
        # vol_params: [D1, D2, D3]
        if dev_params is None:
            dev_params = [0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        if vol_params is None:
            vol_params = [1.5, 0.0, 0.0]
        self.dev_params = dev_params
        self.vol_params = vol_params

    def _eval_terms(self, F: jnp.ndarray):
        if F.shape[-2:] == (2, 2):
            F = jnp.array([[F[0, 0], F[0, 1], 0.], 
                           [F[1, 0], F[1, 1], 0.],
                           [0.,      0.,     1. ]])
        C = C_func(F)
        I1 = I1_func(C)
        I2 = I2_func(C)
        I3 = I3_func(C)
        I3_safe = jnp.clip(I3, 1.0e-8, 1.0e8)
        i1_dev = I3_safe**(-1/3) * I1
        i2_dev = I3_safe**(-2/3) * I2

        X = i1_dev - 3.0
        Y = i2_dev - 3.0
        
        # Logarithmic Terms
        log1 = jnp.log(jnp.clip(i1_dev / 3.0, 1.0e-8, 1.0e8))
        log2 = jnp.log(jnp.clip(i2_dev / 3.0, 1.0e-8, 1.0e8))

        # Deviatoric Terms (Order 3 + Logarithmic Terms)
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
            (self.dev_params[9] * log2 if len(self.dev_params) > 9 else 0.0)
        )
        
        J = jnp.sqrt(I3_safe)
        J_minus_1 = J - 1.0

        # Volumetric Terms (Order 3)
        vol_terms = (
            self.vol_params[0] * J_minus_1**2 + 
            self.vol_params[1] * J_minus_1**4 +
            self.vol_params[2] * J_minus_1**6
        )
        
        return dev_terms, vol_terms

    def psi_dev(self, F: jnp.ndarray) -> jnp.ndarray:
        dev_terms, _ = self._eval_terms(F)
        return dev_terms

    def psi_vol(self, F: jnp.ndarray) -> jnp.ndarray:
        _, vol_terms = self._eval_terms(F)
        return vol_terms

    def psi(self, F: jnp.ndarray) -> jnp.ndarray:
        dev_terms, vol_terms = self._eval_terms(F)
        return dev_terms + vol_terms


@register_material("gentthomas")
class GentThomas(GeneralizedMooneyRivlin):
    def __init__(self, c1=0.5, c2=1.5, jit_P: bool = True):
        # Gent-Thomas derived as a special case of GMR:
        # C10 = c1 (default 0.5), E = 1.0, D1 = c2 (default 1.5)
        dev_params = [c1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        vol_params = [c2, 0.0, 0.0]
        super().__init__(dev_params=dev_params, vol_params=vol_params, jit_P=jit_P)


@register_material("nh4")
@register_material("neohookean4")
class NeoHookean4(GeneralizedMooneyRivlin):
    def __init__(self, c10=0.5, d2=1.5, jit_P: bool = True):
        # NeoHookean4 (psi = c10*(I1-3) + d2*(J-1)^4) derived as a special case of GMR:
        # C10 = c10 (default 0.5), D2 = d2 (default 1.5)
        dev_params = [c10, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        vol_params = [0.0, d2, 0.0]
        super().__init__(dev_params=dev_params, vol_params=vol_params, jit_P=jit_P)


@register_material("nh2")
@register_material("neohookean2")
@register_material("nh")
class NeoHookeanGMR(GeneralizedMooneyRivlin):
    def __init__(self, c10=0.5, d1=1.5, jit_P: bool = True):
        # NeoHookean2 (psi = c10*(I1-3) + d1*(J-1)^2) derived as a special case of GMR:
        # C10 = c10 (default 0.5), D1 = d1 (default 1.5)
        dev_params = [c10, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        vol_params = [d1, 0.0, 0.0]
        super().__init__(dev_params=dev_params, vol_params=vol_params, jit_P=jit_P)



@register_material("c20_d10_d05")
@register_material("c20d10d05")
class CustomGT(GeneralizedMooneyRivlin):
    def __init__(self, c10=2.0, d1=1.0, d2=0.5, jit_P: bool = True):
        # Custom Ground Truth Model: C10=2.0, D1=1.0, D2=0.5
        dev_params = [c10, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        vol_params = [d1, d2, 0.0]
        super().__init__(dev_params=dev_params, vol_params=vol_params, jit_P=jit_P)

@register_material("aniso30")
class Aniso30(BaseMaterialModel):
    def __init__(self, c10=0.5, d1=1.0, theta=jnp.pi/6.0, jit_P: bool = True):
        super().__init__(jit_P=jit_P)
        self.c10 = c10
        self.d1 = d1
        # Calculate fiber direction vector from angle (theta in XY plane)
        self.a0 = jnp.array([jnp.cos(theta), jnp.sin(theta), 0.0])

    def psi(self, F: jnp.ndarray) -> jnp.ndarray:
        return self.psi_dev(F) + self.psi_vol(F) + self.psi_aniso(F)

    def psi_dev(self, F: jnp.ndarray) -> jnp.ndarray:
        if F.shape[-2:] == (2, 2):
            F = jnp.array([[F[0, 0], F[0, 1], 0.], 
                           [F[1, 0], F[1, 1], 0.],
                           [0.,      0.,     1. ]])
        C = C_func(F)
        I3_safe = jnp.clip(I3_func(C), 1.0e-8, 1.0e8)
        C_bar = (I3_safe**(-1/3))[..., None, None] * C
        return self.c10 * (I1_func(C_bar) - 3.0)

    def psi_vol(self, F: jnp.ndarray) -> jnp.ndarray:
        if F.shape[-2:] == (2, 2):
            F = jnp.array([[F[0, 0], F[0, 1], 0.], 
                           [F[1, 0], F[1, 1], 0.],
                           [0.,      0.,     1. ]])
        C = C_func(F)
        I3_safe = jnp.clip(I3_func(C), 1.0e-8, 1.0e8)
        return self.d1 * (jnp.sqrt(I3_safe) - 1.0)**2

    def psi_aniso(self, F: jnp.ndarray) -> jnp.ndarray:
        if F.shape[-2:] == (2, 2):
            F = jnp.array([[F[0, 0], F[0, 1], 0.], 
                           [F[1, 0], F[1, 1], 0.],
                           [0.,      0.,     1. ]])
        C = C_func(F)
        I3_safe = jnp.clip(I3_func(C), 1.0e-8, 1.0e8)
        C_bar = (I3_safe**(-1/3))[..., None, None] * C
        I4_bar = jnp.einsum('i,...ij,j->...', self.a0, C_bar, self.a0)
        return 0.7 * (I4_bar - 1.0)**2


@register_material("ortho45")
class Ortho45(BaseMaterialModel):
    def __init__(self, c10=0.5, d1=1.0, theta1=jnp.pi/4.0, theta2=-jnp.pi/4.0, jit_P: bool = True):
        super().__init__(jit_P=jit_P)
        self.c10 = c10
        self.d1 = d1
        self.a1 = jnp.array([jnp.cos(theta1), jnp.sin(theta1), 0.0])
        self.a2 = jnp.array([jnp.cos(theta2), jnp.sin(theta2), 0.0])

    def psi(self, F: jnp.ndarray) -> jnp.ndarray:
        return self.psi_dev(F) + self.psi_vol(F) + self.psi_aniso(F)

    def psi_dev(self, F: jnp.ndarray) -> jnp.ndarray:
        if F.shape[-2:] == (2, 2):
            F = jnp.array([[F[0, 0], F[0, 1], 0.], 
                           [F[1, 0], F[1, 1], 0.],
                           [0.,      0.,     1. ]])
        C = C_func(F)
        I3_safe = jnp.clip(I3_func(C), 1.0e-8, 1.0e8)
        C_bar = (I3_safe**(-1/3))[..., None, None] * C
        return self.c10 * (I1_func(C_bar) - 3.0)

    def psi_vol(self, F: jnp.ndarray) -> jnp.ndarray:
        if F.shape[-2:] == (2, 2):
            F = jnp.array([[F[0, 0], F[0, 1], 0.], 
                           [F[1, 0], F[1, 1], 0.],
                           [0.,      0.,     1. ]])
        C = C_func(F)
        I3_safe = jnp.clip(I3_func(C), 1.0e-8, 1.0e8)
        return self.d1 * (jnp.sqrt(I3_safe) - 1.0)**2

    def psi_aniso(self, F: jnp.ndarray) -> jnp.ndarray:
        if F.shape[-2:] == (2, 2):
            F = jnp.array([[F[0, 0], F[0, 1], 0.], 
                           [F[1, 0], F[1, 1], 0.],
                           [0.,      0.,     1. ]])
        C = C_func(F)
        I3_safe = jnp.clip(I3_func(C), 1.0e-8, 1.0e8)
        C_bar = (I3_safe**(-1/3))[..., None, None] * C
        I4_bar_1 = jnp.einsum('i,...ij,j->...', self.a1, C_bar, self.a1)
        I4_bar_2 = jnp.einsum('i,...ij,j->...', self.a2, C_bar, self.a2)
        return 0.7 * (I4_bar_1 - 1.0)**2 + 0.9 * (I4_bar_2 - 1.0)**2



@register_material("ogden")
class Ogden(BaseMaterialModel):
    def __init__(self, mu_params, alpha_params, vol_params, jit_P: bool = True):
        super().__init__(jit_P=jit_P)
        # mu_params: [mu1, mu2, mu3]
        # alpha_params: [alpha1, alpha2, alpha3]
        # vol_params: [D1, D2, D3]
        self.mu_params = mu_params
        self.alpha_params = alpha_params
        self.vol_params = vol_params

    def psi(self, F: jnp.ndarray) -> jnp.ndarray:
        if F.shape[-2:] == (2, 2):
            F = jnp.array([[F[0, 0], F[0, 1], 0.], 
                           [F[1, 0], F[1, 1], 0.],
                           [0.,      0.,     1. ]])
        C = C_func(F)
        I3 = I3_func(C)
        I3_safe = jnp.clip(I3, 1.0e-8, 1.0e8)
        
        # Calculate eigenvalues of C to get lambda^2
        # Use eigvalsh as C is symmetric
        eigenvalues = jnp.linalg.eigvalsh(C)
        eigenvalues = jnp.clip(eigenvalues, 1.0e-8, 1.0e8)
        lam = jnp.sqrt(eigenvalues)
        
        # Deviatoric stretches
        lam_bar = lam * I3_safe**(-1/6)
        
        dev_terms = 0.0
        for i in range(3):
            # To avoid numerical issues when alpha -> 0 or lam_bar -> 0
            val = (lam_bar[0]**self.alpha_params[i] + lam_bar[1]**self.alpha_params[i] + lam_bar[2]**self.alpha_params[i] - 3.0)
            # Use jnp.where or similar if alpha can exactly be zero, but we assume alpha != 0
            alpha_safe = jnp.where(jnp.abs(self.alpha_params[i]) < 1e-6, 1e-6 * jnp.sign(self.alpha_params[i] + 1e-9), self.alpha_params[i])
            dev_terms += (self.mu_params[i] / alpha_safe) * val
            
        J = jnp.sqrt(I3_safe)
        J_minus_1 = J - 1.0

        vol_terms = (
            self.vol_params[0] * J_minus_1**2 + 
            self.vol_params[1] * J_minus_1**4 +
            self.vol_params[2] * J_minus_1**6
        )
        
        return dev_terms + vol_terms

@register_material("holzapfel")
class AnisotropicHolzapfel(BaseMaterialModel):
    """
    Anisotropic Holzapfel-Gasser-Ogden material model with 1 fiber orientation.
    Uses Macauley bracket to ensure fibers do not support compression.
    """
    def __init__(self, c10=0.5, d1=1.5, k1=0.8, k2=0.9, kappa=0.0, theta=jnp.pi/6.0, jit_P: bool = True):
        super().__init__(jit_P=jit_P)
        self.c10 = c10
        self.d1 = d1
        self.k1 = k1
        self.k2 = k2
        self.kappa = kappa
        
        # Calculate fiber direction vector from angle (theta in XY plane)
        self.a0 = jnp.array([jnp.cos(theta), jnp.sin(theta), 0.0])

    def psi(self, F: jnp.ndarray) -> jnp.ndarray:
        return self.psi_dev(F) + self.psi_vol(F) + self.psi_aniso(F)

    def psi_dev(self, F: jnp.ndarray) -> jnp.ndarray:
        if F.shape[-2:] == (2, 2):
            F = jnp.array([[F[0, 0], F[0, 1], 0.], 
                           [F[1, 0], F[1, 1], 0.],
                           [0.,      0.,     1. ]])
        C = C_func(F)
        I3_safe = jnp.clip(I3_func(C), 1.0e-8, 1.0e8)
        C_bar = (I3_safe**(-1/3))[..., None, None] * C
        return self.c10 * (I1_func(C_bar) - 3.0)

    def psi_vol(self, F: jnp.ndarray) -> jnp.ndarray:
        if F.shape[-2:] == (2, 2):
            F = jnp.array([[F[0, 0], F[0, 1], 0.], 
                           [F[1, 0], F[1, 1], 0.],
                           [0.,      0.,     1. ]])
        C = C_func(F)
        I3_safe = jnp.clip(I3_func(C), 1.0e-8, 1.0e8)
        return self.d1 * (jnp.sqrt(I3_safe) - 1.0)**2

    def psi_aniso(self, F: jnp.ndarray) -> jnp.ndarray:
        if F.shape[-2:] == (2, 2):
            F = jnp.array([[F[0, 0], F[0, 1], 0.], 
                           [F[1, 0], F[1, 1], 0.],
                           [0.,      0.,     1. ]])
        C = C_func(F)
        I3_safe = jnp.clip(I3_func(C), 1.0e-8, 1.0e8)
        C_bar = (I3_safe**(-1/3))[..., None, None] * C
        I1_bar = I1_func(C_bar)
        I4_bar = jnp.einsum('i,...ij,j->...', self.a0, C_bar, self.a0)
        E_bar = self.kappa * (I1_bar - 3.0) + (1.0 - 3.0 * self.kappa) * (I4_bar - 1.0)
        E_active = jnp.maximum(E_bar, 0.0)
        return (self.k1 / (2.0 * self.k2)) * (jnp.exp(self.k2 * (E_active**2)) - 1.0)

