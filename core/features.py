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
    def __init__(self, a0: jnp.ndarray, cap_compression: bool = True):
        self.a0 = jnp.asarray(a0, dtype=jnp.float64)
        self.cap_compression = cap_compression

    def extract(self, f: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Extracts dev, vol, and aniso features.
        aniso is based on the isochoric I4_bar.
        Returns: (dev, vol, aniso)
        """
        invariants, _ = invariants_and_derivatives(f)
        dev, vol = transform_input_features(invariants)
        
        C = C_func(f)
        I3 = jnp.clip(I3_func(C), 1.0e-8, 1.0e8)
        C_bar = (I3**(-1/3))[..., None, None] * C
        I4_bar = I4_func(C_bar, self.a0)
        I5_bar = I5_func(C_bar, self.a0)
        
        I4_val = I4_bar - 1.0
        I5_val = I5_bar - 1.0
        
        # Optionally cap at 0.0 to enforce that fibers do not resist compression.
        if self.cap_compression:
            I4_val = jnp.maximum(I4_val, 0.0)
            I5_val = jnp.maximum(I5_val, 0.0)
            
        aniso = jnp.stack([I4_val, I5_val], axis=-1)
        
        return dev, vol, aniso
