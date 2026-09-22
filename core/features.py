import jax
import jax.numpy as jnp
from abc import ABC, abstractmethod
from typing import Tuple
from .utils import invariants_and_derivatives, transform_input_features, C_func, I3_func, I4_func, I5_func

class FeatureExtractor(ABC):
    @abstractmethod
    def extract(self, f: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Extracts features (e.g., invariants) from the deformation gradient F.
        Returns:
            Tuple of features to be fed into the GP components.
            Currently expects (dev_features, vol_features).
        """
        pass

class IsotropicFeatureExtractor(FeatureExtractor):
    def extract(self, f: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Extracts standard isotropic invariants (I1, I2, J) and splits them
        into deviatoric and volumetric features.
        """
        invariants, _ = invariants_and_derivatives(f)
        dev, vol = transform_input_features(invariants)
        return dev, vol

class AnisotropicFeatureExtractor(FeatureExtractor):
    def __init__(self, a0: jnp.ndarray, a1: jnp.ndarray = None, cap_compression: bool = False):
        self.a0 = jnp.asarray(a0, dtype=jnp.float64)
        if a1 is not None:
            self.a1 = jnp.asarray(a1, dtype=jnp.float64)
        else:
            # Default to in-plane orthogonal direction a1 = a0_perp: [-a0_y, a0_x, 0.0]
            # This ensures orthogonality (a0 . a1 = 0 => I8 = 0) in 1-fiber or orthotropic cases
            self.a1 = jnp.array([-self.a0[1], self.a0[0], 0.0], dtype=jnp.float64)
        self.cap_compression = cap_compression

    def extract(self, f: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Extracts dev, vol, and aniso features.
        aniso is always a 3-dimensional vector [I4_bar, I6_bar, I8_bar]:
          - I4_bar = a0 . C_bar . a0
          - I6_bar = a1 . C_bar . a1
          - I8_bar = (a0 . a1) * (a0 . C_bar . a1)
        When only 1 fiber family is present (a1 = a0_perp), a0 . a1 = 0 => I8_bar = 0,
        and ARD pruning naturally flattens lengthscales for I6_bar and I8_bar.
        Returns: (dev, vol, aniso)
        """
        invariants, _ = invariants_and_derivatives(f)
        dev, vol = transform_input_features(invariants)
        
        C = C_func(f)
        I3 = jnp.clip(I3_func(C), 1.0e-8, 1.0e8)
        C_bar = (I3**(-1/3))[..., None, None] * C
        
        I4_bar = I4_func(C_bar, self.a0)
        I6_bar = I4_func(C_bar, self.a1)
        
        if self.cap_compression:
            I4_bar = jnp.maximum(I4_bar, 1.0)
            I6_bar = jnp.maximum(I6_bar, 1.0)

        dot_a0_a1 = jnp.dot(self.a0, self.a1)
        C_bar_a1 = jnp.einsum('...ij,j->...i', C_bar, self.a1)
        I8_bar = dot_a0_a1 * jnp.einsum('i,...i->...', self.a0, C_bar_a1)
        
        aniso = jnp.stack([I4_bar, I6_bar, I8_bar], axis=-1)
        
        return dev, vol, aniso

