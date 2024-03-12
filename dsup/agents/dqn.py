import dataclasses
import functools
from typing import TypeVar

import chex
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from clu import metrics as clu_metrics

from dsup.models import NeuralNet, QLearningModel
from dsup.train_state import FittedValueTrainState, TargetParamsUpdate
from dsup.trainer import Trainer
from dsup.types import TransitionBatch
from dsup.utils import jitpp
from dsup.utils.jitpp import Bind, Donate, Static
from dsup.utils.schedules import ContinuousTimeSchedule

ObsT = TypeVar("ObsT")
ActT = TypeVar("ActT")

Q_LOSS = "loss__q"
ACTION_PROB_MAX = "debug__action_prob_max"
ACTION_PROB_MIN = "debug__action_prob_min"
SEES_TERMINAL = "debug__sees_terminal"
RATIO_TERMINAL = "debug__ratio_terminal"


@dataclasses.dataclass(frozen=True, kw_only=True)
class DQNTrainer(Trainer[FittedValueTrainState]):
    """Implementation of DQN.

    All `_model` arguments are not the full neural network models, just the backbones.

    Args:
        q_model: The backbone of the network representing the Q-function.
        optim: The optimizer for all neural networks.
        target_params_update: The protocol for updating target parameters.
        h: The timestep length in the MDP.
        discount: The nominal discount factor per unit of time.
        exploration_schedule: An annealing schedule for exploration.
    """

    q_model: nn.Module
    optim: optax.GradientTransformation
    target_params_update: TargetParamsUpdate
    discount: float
    exploration_schedule: ContinuousTimeSchedule[chex.Scalar]

    @functools.cached_property
    def model(self) -> nn.Module:
        return QLearningModel(self.q_model, self.env.action_space.n)

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
            Q_LOSS: clu_metrics.Average.from_output(Q_LOSS),
            ACTION_PROB_MAX: clu_metrics.Average.from_output(ACTION_PROB_MAX),
            ACTION_PROB_MIN: clu_metrics.Average.from_output(ACTION_PROB_MIN),
            SEES_TERMINAL: clu_metrics.Average.from_output(SEES_TERMINAL),
            RATIO_TERMINAL: clu_metrics.Average.from_output(RATIO_TERMINAL),
        }
        return clu_metrics.Collection.create(**metrics)

    @jitpp.jit
    @staticmethod
    @chex.assert_max_traces(2)
    def train_step(
        rng: chex.PRNGKey,
        state: Donate[FittedValueTrainState],
        batch: TransitionBatch,
        *,
        decision_frequency: Bind[int],
        discount: Bind[float],
        num_actions: Bind[Static[int]],
    ) -> FittedValueTrainState:
        h = 1.0 / decision_frequency

        def q_loss(params: chex.ArrayTree, trans: TransitionBatch) -> chex.Scalar:
            q_a_tm1 = state.apply_fn(params, trans.o_tm1)
            q_tm1 = q_a_tm1[trans.a_tm1]
            q_a_t = state.apply_fn(state.target_params, trans.o_t)
            q_t = jnp.max(q_a_t)

            gamma = discount**h
            # at episode end, reward is atomic
            # otherwise, reward is a rate
            r_weight = 1.0 + (h - 1.0) * trans.m_t

            q_target = r_weight * trans.r_tm1 + gamma * trans.m_t * q_t

            loss = 0.5 * (q_target - q_tm1) ** 2
            return loss

        @jax.value_and_grad
        def q_grad_fn(params: chex.ArrayTree, trans: TransitionBatch) -> chex.Scalar:
            return jnp.mean(jax.vmap(q_loss, in_axes=(None, 0))(params, trans))

        loss, grads = q_grad_fn(state.params, batch)

        actions_one_hot = jnp.eye(num_actions)[batch.a_tm1]
        action_probs = jnp.mean(actions_one_hot, axis=0)
        action_prob_max = jnp.max(action_probs)
        action_prob_min = jnp.min(action_probs)

        infos = {
            Q_LOSS: loss,
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
        exploration_schedule: Bind[Static[ContinuousTimeSchedule[chex.Scalar]]],
        train: Static[bool] = True,
    ):
        q_a = state.apply_fn(state.params, obs)
        epsilon = exploration_schedule(state.step)
        q_a = jax.lax.cond(
            train and (jax.random.bernoulli(rng, epsilon) == 1.0),
            lambda: jax.random.uniform(rng, shape=q_a.shape),
            lambda: q_a,
        )
        return jnp.argmax(q_a)
