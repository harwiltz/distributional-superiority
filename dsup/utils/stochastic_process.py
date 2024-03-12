import abc
import dataclasses
from typing import Generic, TypeVar

import chex
import jax
import jax.numpy as jnp
from typing_extensions import Self

T = TypeVar("T")


class StochasticProcess(Generic[T], abc.ABC):
    @abc.abstractmethod
    def step(self: Self, rng: chex.PRNGKey, x: T, h: float) -> T: ...


@dataclasses.dataclass(frozen=True, kw_only=True)
class OrnsteinUhlenbeckProcess(StochasticProcess[jax.Array]):
    stiffness: float
    scale: float

    def step(self: Self, rng: chex.PRNGKey, x: jax.Array, h: float) -> jax.Array:
        drift = -x * self.stiffness
        noise = self.scale * jax.random.normal(rng, shape=x.shape)
        return x + drift * h + noise * jnp.sqrt(h)


@dataclasses.dataclass(frozen=True, kw_only=True)
class WhiteNoiseProcess(StochasticProcess[jax.Array]):
    scale: float

    def step(self: Self, rng: chex.PRNGKey, x: jax.Array, h: float) -> jax.Array:
        return self.scale * jax.random.normal(rng, shape=x.shape)


@dataclasses.dataclass(frozen=True, kw_only=True)
class BrownianMotionProcess(StochasticProcess[jax.Array]):
    scale: float = 1.0

    def step(self: Self, rng: chex.PRNGKey, x: jax.Array, h: float) -> jax.Array:
        noise = self.scale * jax.random.normal(rng, x.shape)
        return x + noise * jnp.sqrt(h)


@dataclasses.dataclass(frozen=True, kw_only=True)
class EpsilonGreedyProcess(StochasticProcess[bool]):
    epsilon: float

    def step(self: Self, rng: chex.PRNGKey, x: jax.Array, h: float) -> jax.Array:
        del h
        return jax.random.bernoulli(rng, self.epsilon, shape=x.shape)


@dataclasses.dataclass(frozen=True, kw_only=True)
class ZeroProcess(StochasticProcess[jax.Array]):
    def step(self: Self, rng: chex.PRNGKey, x: jax.Array, h: float) -> jax.Array:
        del rng
        del h
        return 0.0
