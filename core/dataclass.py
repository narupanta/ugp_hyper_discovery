"""Mean and variance containers returned by the GP teachers (core/hsgp_model.py, core/exact_gp.py)."""
from typing import NamedTuple
import jax.numpy as jnp


class EnergyDist(NamedTuple):
    mean: jnp.ndarray
    var: jnp.ndarray


class StressDist(NamedTuple):
    mean: jnp.ndarray
    var: jnp.ndarray
