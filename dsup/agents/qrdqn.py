import dataclasses
import functools
from typing import TypeVar

import chex
import flax.linen as nn
import gymnasium as gym
import jax
import jax.numpy as jnp
import optax
from clu import metrics as clu_metrics

from dsup.losses import quantile_huber_loss
from dsup.models import ActionConditionedHead, NeuralNet, QuantileModel
from dsup.statistical_functionals import SampleStatisticalFunctional
from dsup.train_state import FittedValueTrainState, TargetParamsUpdate
from dsup.trainer import Trainer
from dsup.types import TransitionBatch
from dsup.utils import jitpp
from dsup.utils.jitpp import Bind, Donate, Static
from dsup.utils.schedules import ContinuousTimeSchedule

ObsT = TypeVar("ObsT")
ActT = TypeVar("ActT")

ZETA_LOSS = "loss__zeta"
ACTION_PROB_MAX = "debug__action_prob_max"
ACTION_PROB_MIN = "debug__action_prob_min"
SEES_TERMINAL = "debug__sees_terminal"
RATIO_TERMINAL = "debug__ratio_terminal"


@dataclasses.dataclass(frozen=True, kw_only=True)
class QRDQNTrainer(Trainer[FittedValueTrainState]):
    """Implementation of QR-DQN, as described by Dabney et. al (https://arxiv.org/abs/1710.10044).

    All `_model` arguments are not the full neural network models, just the backbones.

    Args:
        quantile_model: The backbone of the network representing the quantiles of the return distribution function.
        num_atoms: The number of quantiles to model for each return distribution.
        optim: The optimizer for all neural networks.
        target_params_update: The protocol for updating target parameters.
        statistical_functional: The statistical functional used to rank actions by their return distributions.
        h: The timestep length in the MDP.
        discount: The nominal discount factor per unit of time.
        kappa: The threshold parameter for the quantile huber loss.
        exploration_schedule: An annealing schedule for exploration.
    """

    quantile_model: nn.Module
    num_atoms: int
    optim: optax.GradientTransformation
    target_params_update: TargetParamsUpdate
    discount: float
    kappa: float
    statistical_functional: SampleStatisticalFunctional
    exploration_schedule: ContinuousTimeSchedule[chex.Scalar]

    @property
    def identifier(self) -> str:
        return "QRDQN"

    @functools.cached_property
    def model(self) -> nn.Module:
        return QuantileModel(
            self.quantile_model,
            self.env.action_space.n,
            self.num_atoms,
            risk_measure=self.statistical_functional,
        )

    @functools.cached_property
    def state(self) -> FittedValueTrainState:
        rng, init_rng = jax.random.split(jax.random.PRNGKey(self.seed))
        dummy_data = jnp.array(self.env.observation_space.sample())
        params = self.model.init(init_rng, dummy_data)
        return FittedValueTrainState.create(
            params=params,
            target_params_update=self.target_params_update,
            apply_fn=self.model.apply,
            tx=self.optim,
            metrics=self.metrics.empty(),
        )

    @functools.cached_property
    def metrics(self) -> type[clu_metrics.Collection]:
        metrics = {
            ZETA_LOSS: clu_metrics.Average.from_output(ZETA_LOSS),
            ACTION_PROB_MAX: clu_metrics.Average.from_output(ACTION_PROB_MAX),
            ACTION_PROB_MIN: clu_metrics.Average.from_output(ACTION_PROB_MIN),
            SEES_TERMINAL: clu_metrics.Average.from_output(SEES_TERMINAL),
            RATIO_TERMINAL: clu_metrics.Average.from_output(RATIO_TERMINAL),
        }
        return clu_metrics.Collection.create(**metrics)

    def decision_dists(
        self, state: FittedValueTrainState, obs: chex.Array
    ) -> chex.Array:
        return state.apply_fn(state.params, obs)

    @jitpp.jit
    @staticmethod
    @chex.assert_max_traces(2)
    def train_step(
        rng: chex.PRNGKey,
        state: Donate[FittedValueTrainState],
        batch: TransitionBatch,
        *,
        statistical_functional: Bind[Static[SampleStatisticalFunctional]],
        num_actions: Bind[Static[int]],
        decision_frequency: Bind[int],
        discount: Bind[float],
        kappa: Bind[float],
    ) -> FittedValueTrainState:
        h = 1.0 / decision_frequency

        def quantile_loss(
            params: chex.ArrayTree, trans: TransitionBatch
        ) -> chex.Scalar:
            zeta_a_tm1 = state.apply_fn(params, trans.o_tm1)
            zeta_tm1 = zeta_a_tm1[trans.a_tm1, :]
            zeta_a_t = state.apply_fn(state.target_params, trans.o_t)
            utility_a_t = jax.vmap(statistical_functional.evaluate)(zeta_a_t)
            a_t = jnp.argmax(utility_a_t)
            eta_t = zeta_a_t[a_t, :]

            gamma = discount**h
            # at episode end, reward is atomic
            # otherwise, reward is a rate
            r_weight = 1.0 + (h - 1.0) * trans.m_t

            zeta_target = r_weight * trans.r_tm1 + gamma * trans.m_t * eta_t
            loss = quantile_huber_loss(zeta_target, zeta_tm1, kappa=kappa)
            return loss

        @jax.value_and_grad
        def quantile_grad_fn(
            params: chex.ArrayTree, trans: TransitionBatch
        ) -> chex.Scalar:
            return jnp.mean(jax.vmap(quantile_loss, in_axes=(None, 0))(params, trans))

        loss, grads = quantile_grad_fn(state.params, batch)

        actions_one_hot = jnp.eye(num_actions)[batch.a_tm1]
        action_probs = jnp.mean(actions_one_hot, axis=0)
        action_prob_max = jnp.max(action_probs)
        action_prob_min = jnp.min(action_probs)
        infos = {
            ZETA_LOSS: loss,
            ACTION_PROB_MAX: action_prob_max,
            ACTION_PROB_MIN: action_prob_min,
            SEES_TERMINAL: jnp.any(batch.m_t == 0.0),
            RATIO_TERMINAL: jnp.mean(1 - batch.m_t),
        }

        state = state.apply_gradients(
            grads=grads,
            metrics=state.metrics.single_from_model_output(**infos),
        )
        return state

    @jitpp.jit
    @staticmethod
    @chex.assert_max_traces(4)
    def action(
        rng: jax.random.PRNGKey,
        state: FittedValueTrainState,
        obs: chex.Array,
        *,
        statistical_functional: Bind[Static[SampleStatisticalFunctional]],
        exploration_schedule: Bind[Static[ContinuousTimeSchedule[chex.Scalar]]],
        train: Static[bool] = True,
    ):
        zeta_a = state.apply_fn(state.params, obs)
        stats_a = jax.vmap(statistical_functional.evaluate)(zeta_a)
        epsilon = exploration_schedule(state.step)
        stats_a = jax.lax.cond(
            train and (jax.random.bernoulli(rng, epsilon) == 1.0),
            lambda: jax.random.uniform(rng, shape=stats_a.shape),
            lambda: stats_a,
        )
        return jnp.argmax(stats_a)
