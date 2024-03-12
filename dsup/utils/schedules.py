import abc
import dataclasses
from typing import Generic, TypeVar

import jax
import jax.numpy as jnp

T = TypeVar("T")


class ContinuousTimeSchedule(Generic[T], abc.ABC):
    @abc.abstractmethod
    def __call__(self, t: float) -> T: ...


@dataclasses.dataclass(frozen=True, kw_only=True)
class ConstantSchedule(ContinuousTimeSchedule[T]):
    value: T

    def __call__(self, t: float) -> T:
        return self.value


@dataclasses.dataclass(frozen=True, kw_only=True)
class ExponentialDecaySchedule(ContinuousTimeSchedule[T]):
    init_value: T
    decay_rate: float
    min_value: float = 0.0

    def __call__(self, t: float) -> T:
        return max((self.decay_rate**t) * self.init_value, self.min_value)


@dataclasses.dataclass(frozen=True, kw_only=True)
class LinearSchedule(ContinuousTimeSchedule[T]):
    init_value: T
    end_value: T
    horizon: float

    def __call__(self, t: float) -> T:
        progress = jnp.clip(t / self.horizon, 0.0, 1.0)
        return self.init_value + progress * (self.end_value - self.init_value)


@dataclasses.dataclass(frozen=True, kw_only=True)
class CosineDecaySchedule(ContinuousTimeSchedule[T]):
    init_value: T
    horizon: float
    alpha: float = 0.0
    exponent: float = 1.0

    def __call__(self, t: float) -> T:
        coeff = self.init_value * (1.0 - self.alpha) / 2
        cos_factor = 1.0 + jnp.cos(jnp.pi * t / self.horizon) ** self.exponent
        return coeff * cos_factor + self.alpha
