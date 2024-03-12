import dataclasses
from typing import Callable, Protocol

import jax
import jax.numpy as jnp


class SampleStatisticalFunctional(Protocol):
    def evaluate(self, samples: jax.Array) -> float: ...


@dataclasses.dataclass(frozen=True)
class MeanFunctional:
    def evaluate(self, samples: jax.Array) -> float:
        return jnp.mean(samples)


@dataclasses.dataclass(frozen=True)
class MeanVarianceFunctional:
    var_penalty: float

    def evaluate(self, samples: jax.Array) -> float:
        return jnp.mean(samples) - self.var_penalty * jnp.var(samples)


@dataclasses.dataclass(frozen=True, kw_only=True)
class DistortionRiskFunctional:
    risk_aversion_fn: Callable[[float], float]
    requires_sort: bool = False

    def evaluate(self, samples: jax.Array) -> float:
        samples = jax.lax.cond(self.requires_sort, jnp.sort, lambda x: x, samples)
        n_atoms = samples.shape[-1]
        midpoints = (jnp.arange(n_atoms) + 0.5) / n_atoms
        weights = jax.vmap(self.risk_aversion_fn)(midpoints)
        return jnp.dot(weights, samples) / n_atoms


def CVaRFunctional(alpha: float, requires_sort: bool = False):
    return DistortionRiskFunctional(
        risk_aversion_fn=lambda t: (t < alpha) / alpha, requires_sort=requires_sort
    )
