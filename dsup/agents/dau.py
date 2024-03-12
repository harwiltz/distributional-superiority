import dataclasses
import functools
from typing import TypeVar

import chex
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from clu import metrics as clu_metrics

from dsup.models import AdvantageModel, NeuralNet
from dsup.train_state import ExploratoryTrainState, TargetParamsUpdate
from dsup.trainer import Trainer
from dsup.types import TransitionBatch
from dsup.utils import jitpp
from dsup.utils.jitpp import Bind, Donate, Static
from dsup.utils.schedules import ContinuousTimeSchedule
from dsup.utils.stochastic_process import StochasticProcess

ObsT = TypeVar("ObsT")
ActT = TypeVar("ActT")

ADVANTAGE_LOSS = "loss__advantage"
ADVANTAGE_SCALE = "debug__advantage_scale"
VALUE_SCALE = "debug__value_scale"
ACTION_GAP = "info__action_gap"
NOISE_SCALE = "debug__noise_scale"
ACTION_PROB_MAX = "debug__action_prob_max"
ACTION_PROB_MIN = "debug__action_prob_min"
SEES_TERMINAL = "debug__sees_terminal"
RATIO_TERMINAL = "debug__ratio_terminal"


class DAUModel(nn.Module):
    advantage_model: AdvantageModel
    value_model: NeuralNet
    h: float

    def advantage(self, x):
        return self.advantage_model(x)

    def value(self, x):
        return self.value_model(x)

    def q_values(self, x):
        a, v = self(x)
        return v + self.h * a

    def __call__(self, x):
        return self.advantage_model(x), self.value_model(x)

    def act(self, x):
        return jnp.argmax(self.advantage_model(x))


@dataclasses.dataclass(frozen=True, kw_only=True)
class DAUTrainer(Trainer[ExploratoryTrainState]):
    """Implementation of Deep Advantage Updating, as proposed by Tallec et al. (http://arxiv.org/abs/1901.09732)."""

    advantage_model: nn.Module
    value_model: nn.Module
    optim: optax.GradientTransformation
    target_params_update: TargetParamsUpdate
    discount: float
    exploration_process: StochasticProcess[chex.Array]
    exploration_schedule: ContinuousTimeSchedule[chex.Scalar]

    @functools.cached_property
    def metrics(self) -> type[clu_metrics.Collection]:
        scalar_metric_tags = [
            ADVANTAGE_LOSS,
            ADVANTAGE_SCALE,
            VALUE_SCALE,
            NOISE_SCALE,
            ACTION_GAP,
            SEES_TERMINAL,
            RATIO_TERMINAL,
            ACTION_PROB_MAX,
            ACTION_PROB_MIN,
        ]

        scalar_metrics = {
            metric: clu_metrics.Average.from_output(metric)
            for metric in scalar_metric_tags
        }
        return clu_metrics.Collection.create(**scalar_metrics)

    @functools.cached_property
    def model(self) -> DAUModel:
        return DAUModel(
            AdvantageModel(self.advantage_model, self.num_actions),
            NeuralNet(torso=self.value_model, head=nn.Dense(1)),
            1.0 / self.decision_frequency,
        )

    @functools.cached_property
    def state(self) -> ExploratoryTrainState:
        rng, init_rng = jax.random.split(jax.random.PRNGKey(self.seed))
        params = self.model.init(init_rng, self.env.observation_space.sample())
        return ExploratoryTrainState.create(
            params=params,
            target_params_update=self.target_params_update,
            noise_state=jnp.zeros(self.num_actions),
            apply_fn=self.model.apply,
            tx=self.optim,
            metrics=self.metrics.empty(),
        )

    @jitpp.jit
    @staticmethod
    @chex.assert_max_traces(2)
    def train_step(
        rng: chex.PRNGKey,
        state: Donate[ExploratoryTrainState],
        batch: TransitionBatch,
        *,
        decision_frequency: Bind[int],
        discount: Bind[float],
        num_actions: Bind[Static[int]],
        exploration_process: Bind[Static[StochasticProcess[chex.Array]]],
    ) -> ExploratoryTrainState:
        h = 1.0 / decision_frequency

        def loss_fn(params: chex.ArrayTree, trans: TransitionBatch) -> chex.Scalar:
            v_t = jnp.squeeze(
                state.apply_fn(state.target_params, trans.o_t, method="value")
            )
            gamma = discount**h
            # at episode end, reward is atomic
            # otherwise, reward is a rate
            r_weight = 1.0 + (h - 1.0) * trans.m_t
            v_target = r_weight * trans.r_tm1 + gamma * trans.m_t * v_t
            q_a_tm1 = jnp.squeeze(
                state.apply_fn(params, trans.o_tm1, method="q_values")
            )
            q_tm1 = q_a_tm1[trans.a_tm1]
            return 0.5 * (q_tm1 - v_target) ** 2

        @jax.value_and_grad
        def grad_fn(params: chex.ArrayTree, batch: TransitionBatch) -> chex.Scalar:
            return jnp.mean(jax.vmap(loss_fn, in_axes=(None, 0))(params, batch))

        loss, grads = grad_fn(state.params, batch)
        new_noise_state = exploration_process.step(rng, state.noise_state, h)

        actions_one_hot = jnp.eye(num_actions)[batch.a_tm1]
        action_probs = jnp.mean(actions_one_hot, axis=0)
        action_prob_max = jnp.max(action_probs)
        action_prob_min = jnp.min(action_probs)

        advs, vs = jax.vmap(state.apply_fn, in_axes=(None, 0))(
            state.params, batch.o_tm1
        )
        adv_mean = jnp.mean(jnp.abs(advs))
        v_mean = jnp.mean(jnp.abs(vs))

        advs_sorted = jnp.sort(advs, axis=-1, descending=True)
        action_gap = jnp.mean(jnp.abs(advs_sorted[:, 0] - advs_sorted[:, 1]))

        noise_scale = jnp.mean(jnp.abs(state.noise_state))

        infos = {
            ADVANTAGE_LOSS: loss,
            ACTION_PROB_MAX: action_prob_max,
            ACTION_PROB_MIN: action_prob_min,
            SEES_TERMINAL: jnp.any(batch.m_t == 0.0),
            RATIO_TERMINAL: jnp.mean(1 - batch.m_t),
            ADVANTAGE_SCALE: adv_mean,
            VALUE_SCALE: v_mean,
            ACTION_GAP: action_gap,
            NOISE_SCALE: noise_scale,
        }

        return state.apply_gradients(
            grads=grads,
            metrics=state.metrics.single_from_model_output(**infos),
            noise_state=new_noise_state,
        )

    @jitpp.jit
    @staticmethod
    @chex.assert_max_traces(2)
    def action(
        rng: jax.random.PRNGKey,
        state: ExploratoryTrainState,
        obs: chex.Array,
        *,
        exploration_schedule: Bind[Static[ContinuousTimeSchedule[chex.Scalar]]],
        train: Static[bool] = True,
    ):
        adv_a = state.apply_fn(state.params, obs, method="advantage")
        epsilon = exploration_schedule(state.step)
        adv_a = adv_a + train * epsilon * state.noise_state
        return jnp.argmax(adv_a)
