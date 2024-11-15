from typing import Callable

import jax
import jax.numpy as jnp
from flax import linen as nn

from dsup import statistical_functionals as risk


class MLPTorso(nn.Module):
    num_layers: int
    num_hidden_units: int
    module: nn.Module = nn.Dense
    activation: Callable[[jax.Array], jax.Array] = nn.leaky_relu
    num_outputs: int | None = None

    @nn.compact
    def __call__(self, x):
        for _ in range(self.num_layers):
            x = self.module(self.num_hidden_units)(x)
            x = self.activation(x)
        return x


class ActionConditionedHead(nn.Module):
    num_outputs: int
    num_actions: int | None = None

    @nn.compact
    def __call__(self, x, *, num_actions: int | None = None):
        num_actions = nn.merge_param("num_actions", self.num_actions, num_actions)
        outs = nn.Dense(num_actions * self.num_outputs)(x)
        return jnp.squeeze(jnp.reshape(outs, (num_actions, self.num_outputs)))
        # return einops.rearrange(outs, 'outs -> num_actions num_outputs', num_actions=self.num_actions, num_outputs=self.num_outputs)


class NeuralNet(nn.Module):
    torso: nn.Module
    head: nn.Module

    def __call__(self, x):
        return self.head(self.torso(x))


class QLearningModel(nn.Module):
    torso: nn.Module
    num_actions: int

    @nn.compact
    def __call__(self, x):
        return nn.Dense(self.num_actions)(self.torso(x))

    def action_values(self, x):
        return self.__call__(x)

    def act(self, x) -> int:
        q_values = self.action_values(x)
        return jnp.argmax(q_values)


class AdvantageModel(nn.Module):
    torso: nn.Module
    num_actions: int

    @nn.compact
    def __call__(self, x):
        biased_a = nn.Dense(self.num_actions)(self.torso(x))
        return biased_a - jnp.max(biased_a, axis=0)


class QuantileModel(nn.Module):
    torso: nn.Module
    num_actions: int
    num_atoms: int
    risk_measure: risk.SampleStatisticalFunctional = risk.MeanFunctional()

    def action_values(self, x):
        quantiles = self.__call__(x)
        return jax.vmap(self.risk_measure.evaluate)(quantiles)

    def act(self, x):
        risks = self.action_values(x)
        return jnp.argmax(risks)

    @nn.compact
    def __call__(self, x):
        return ActionConditionedHead(self.num_atoms, self.num_actions)(self.torso(x))


class DistributionalSuperiorityModel(nn.Module):
    little_q_model: nn.Module
    superiority_model: nn.Module
    eta_model: nn.Module
    h: float
    risk_measure: risk.SampleStatisticalFunctional = risk.MeanFunctional()
    rescale_factor: float = 0.5
    shift_by_q: bool = True

    def __call__(self, x):
        q = self.little_q_model(x)
        sup_quantiles = self.superiority_model(x)
        a_star = jnp.argmax(  # TODO: stop grad?
            jax.vmap(self.risk_measure.evaluate)(
                self.shift_by_q * (1 - self.h ** (1 - self.rescale_factor)) * q[:, None]
                + sup_quantiles
            )
        )
        q = q - q[a_star]
        sup_quantiles = sup_quantiles - jnp.expand_dims(sup_quantiles[a_star, :], 0)
        eta_quantiles = self.eta_model(x)
        return q, sup_quantiles, eta_quantiles

    def action_values(self, x):
        q, sup, _ = self.__call__(x)
        quantiles = (
            self.shift_by_q * (1 - self.h ** (1 - self.rescale_factor)) * q[:, None]
            + sup
        )
        return jax.vmap(self.risk_measure.evaluate)(quantiles)

    def act(self, x):
        return jnp.argmax(self.action_values(x))
