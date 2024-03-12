import dataclasses
import functools

import chex
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from clu import metrics as clu_metrics

from dsup import statistical_functionals, train_state
from dsup.losses import quantile_huber_loss
from dsup.models import ActionConditionedHead, DistributionalSuperiorityModel, NeuralNet
from dsup.statistical_functionals import SampleStatisticalFunctional
from dsup.train_state import ExploratoryTrainState
from dsup.trainer import Trainer
from dsup.types import LossArtifacts, TransitionBatch
from dsup.utils import jitpp
from dsup.utils.functions import batch_select_actions
from dsup.utils.jitpp import Bind, Donate, Static
from dsup.utils.schedules import ContinuousTimeSchedule
from dsup.utils.stochastic_process import StochasticProcess

LOSS_Q = "loss__q"
LOSS_ETA = "loss__eta"
LOSS_ZETA = "loss__zeta"
ACTION_PROB_MAX = "debug__action_prob_max"
ACTION_PROB_MIN = "debug__action_prob_min"


@dataclasses.dataclass(frozen=True, kw_only=True)
class DistributionalSuperiorityTrainer(Trainer[ExploratoryTrainState]):
    """Implementation of the Distributional Superiority method.

    All `_model` arguments are not the full neural network models, just the backbones.

    Args:
        little_q_model: The backbone of the network representing the little q-function.
        superiority_model: The backbone of the network representing the quantiles of the distributional superiority.
        eta_model: The backbone of the network representing the quantiles of the policy-conditioned return distribution.
        num_atoms: The number of quantiles to model for each return distribution.
        optim: The optimizer for all neural networks.
        target_params_update: The protocol for updating target parameters.
        statistical_functional: The statistical functional used to rank actions by their return distributions.
        h: The timestep length in the MDP.
        discount: The nominal discount factor per unit of time.
        kappa: The threshold parameter for the quantile huber loss.
        exploration_process: A stochastic process used as auxiliary noise for exploration.
        exploration_schedule: An annealing schedule for exploration.
    """

    little_q_model: nn.Module
    superiority_model: nn.Module
    eta_model: nn.Module
    num_atoms: int
    optim: optax.GradientTransformation
    target_params_update: train_state.TargetParamsUpdate
    statistical_functional: statistical_functionals.SampleStatisticalFunctional
    rescale_factor: float
    shift_by_q: bool
    discount: float
    kappa: float
    exploration_process: StochasticProcess[jax.Array]
    exploration_schedule: ContinuousTimeSchedule[float]

    @property
    def identifier(self) -> str:
        shift_tag = "ShiftByLittleQ" if self.shift_by_q else ""
        scale_tag = f"{self.rescale_factor}-Rescaled"
        return f"DSUP_{shift_tag}_{scale_tag}"

    @functools.cached_property
    def model(self) -> DistributionalSuperiorityModel:
        h = 1.0 / self.decision_frequency
        num_actions = self.env.action_space.n

        return DistributionalSuperiorityModel(
            little_q_model=NeuralNet(
                torso=self.little_q_model, head=ActionConditionedHead(1, num_actions)
            ),
            superiority_model=NeuralNet(
                torso=self.superiority_model,
                head=ActionConditionedHead(self.num_atoms, num_actions),
            ),
            eta_model=NeuralNet(torso=self.eta_model, head=nn.Dense(self.num_atoms)),
            h=h,
            risk_measure=self.statistical_functional,
            shift_by_q=self.shift_by_q,
            rescale_factor=self.rescale_factor,
        )

    @functools.cached_property
    def state(self) -> ExploratoryTrainState:
        rng, init_rng = jax.random.split(jax.random.PRNGKey(self.seed))

        def model_init_fn(rngs: chex.PRNGKey) -> chex.ArrayTree:
            dummy_input = jnp.array(self.env.observation_space.sample())
            return self.model.lazy_init(rngs, dummy_input)

        params = model_init_fn(init_rng)

        return ExploratoryTrainState.create(
            params=params,
            apply_fn=self.model.apply,
            tx=self.optim,
            target_params_update=self.target_params_update,
            metrics=self.metrics.empty(),
            noise_state=jnp.zeros(self.env.action_space.n),
        )

    @functools.cached_property
    def metrics(self) -> type[clu_metrics.Collection]:
        metrics = {
            LOSS_Q: clu_metrics.Average.from_output(LOSS_Q),
            LOSS_ZETA: clu_metrics.Average.from_output(LOSS_ZETA),
            # LOSS_ETA: clu_metrics.Average.from_output(LOSS_ETA),
            ACTION_PROB_MAX: clu_metrics.Average.from_output(ACTION_PROB_MAX),
            ACTION_PROB_MIN: clu_metrics.Average.from_output(ACTION_PROB_MIN),
        }
        return clu_metrics.Collection.create(**metrics)

    def decision_dists(
        self, state: ExploratoryTrainState, obs: chex.Array
    ) -> chex.Array:
        q, sup, _ = state.apply_fn(state.params, obs)
        h = 1.0 / self.decision_frequency
        return self.shift_by_q * (1 - h ** (1 - self.rescale_factor)) * q[:, None] + sup

    @jitpp.jit
    @staticmethod
    @chex.assert_max_traces(2)
    def train_step(
        rng: chex.PRNGKey,
        state: Donate[ExploratoryTrainState],
        batch: TransitionBatch,
        *,
        statistical_functional: Bind[Static[SampleStatisticalFunctional]],
        num_actions: Bind[Static[int]],
        exploration_process: Bind[Static[StochasticProcess[jax.Array]]],
        exploration_schedule: Bind[Static[ContinuousTimeSchedule[float]]],
        decision_frequency: Bind[int],
        discount: Bind[float],
        kappa: Bind[float],
        rescale_factor: Bind[float],
        shift_by_q: Bind[Static[bool]],
    ) -> ExploratoryTrainState:
        h = 1.0 / decision_frequency
        _quantile_huber_loss = functools.partial(quantile_huber_loss, kappa=kappa)
        o_tm1, a_tm1, r_tm1, m_t, o_t = batch
        q_a_t, sup_a_t, eta_t = jax.vmap(state.apply_fn, in_axes=(None, 0))(
            state.target_params, o_t
        )
        # action-conditioned superiority at next state
        shifted_sup_a_t = (
            shift_by_q * (1 - h ** (1 - rescale_factor)) * q_a_t[:, :, None] + sup_a_t
        )

        gamma = discount**h

        @functools.partial(jax.grad, has_aux=True)
        def dsup_grad_fn(params: chex.ArrayTree) -> tuple[chex.Scalar, LossArtifacts]:
            q_a_tm1, sup_a_tm1, eta_tm1 = jax.vmap(state.apply_fn, in_axes=(None, 0))(
                params, o_tm1
            )
            # at episode end, reward is atomic
            # otherwise, reward is a rate
            r_weight = 1.0 + (h - 1.0) * m_t
            actions_one_hot = jnp.eye(num_actions)[a_tm1]
            action_probs = jnp.mean(actions_one_hot, axis=0)
            action_prob_max = jnp.max(action_probs)
            action_prob_min = jnp.min(action_probs)
            with jax.named_scope("q_loss"):
                q_tm1 = batch_select_actions(q_a_tm1, a_tm1)
                V_t = jnp.mean(eta_t, axis=-1)
                Q_target = r_weight * r_tm1 + gamma * m_t * V_t
                V_tm1 = jax.lax.stop_gradient(jnp.mean(eta_tm1, axis=-1))
                Q_pred = V_tm1 + h * q_tm1
                bellman_error = 0.5 * jnp.mean(jnp.square(Q_pred - Q_target))
            with jax.named_scope("distributional_loss"):
                sup_tm1 = batch_select_actions(sup_a_tm1, a_tm1)
                eta_target = (r_weight * r_tm1)[:, None] + gamma * m_t[:, None] * eta_t
                eta_pred = (h**rescale_factor) * sup_tm1 + eta_tm1
                zeta_quantile_loss = jnp.mean(
                    jax.vmap(_quantile_huber_loss)(eta_target, eta_pred)
                )
            infos = {
                LOSS_Q: bellman_error,
                LOSS_ZETA: zeta_quantile_loss,
                ACTION_PROB_MAX: action_prob_max,
                ACTION_PROB_MIN: action_prob_min,
            }
            return bellman_error + zeta_quantile_loss, infos

        dsup_grads, dsup_infos = dsup_grad_fn(state.params)
        return state.apply_gradients(
            grads=dsup_grads,
            metrics=state.metrics.single_from_model_output(**dsup_infos),
            noise_state=exploration_schedule(state.step * h)
            * exploration_process.step(rng, state.noise_state, h),
        )

    @jitpp.jit
    @staticmethod
    @chex.assert_max_traces(4)
    def action(
        rng: jax.random.PRNGKey,
        state: ExploratoryTrainState,
        obs: chex.Array,
        *,
        statistical_functional: Bind[Static[SampleStatisticalFunctional]],
        train: Static[bool] = True,
        exploration_schedule: Bind[Static[ContinuousTimeSchedule[chex.Scalar]]],
        shift_by_q: Bind[Static[bool]],
        rescale_factor: Bind[float],
        decision_frequency: Bind[int],
    ):
        h = 1.0 / decision_frequency
        q_a, sup_a, _ = state.apply_fn(state.params, obs)
        quantiles_a = (
            shift_by_q * (1 - h ** (1 - rescale_factor)) * q_a[:, None] + sup_a
        )
        stats_a = jax.vmap(statistical_functional.evaluate)(quantiles_a)
        epsilon = exploration_schedule(state.step)
        stats_a = jax.lax.cond(
            train and (jax.random.bernoulli(rng, epsilon) == 1.0),
            lambda: jax.random.uniform(rng, shape=stats_a.shape),
            lambda: stats_a,
        )
        return jnp.argmax(stats_a)
