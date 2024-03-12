import abc
import dataclasses
import typing
from typing import Any, Callable, Generic, Protocol, TypeVar

import chex
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from flax.training import train_state

ObsT = TypeVar("ObsT")
ActT = TypeVar("ActT")


@dataclasses.dataclass(frozen=True, kw_only=True)
class ContinuousTimeEnvState:
    action_shape: tuple[int] = ()
    done: bool = False
    h: float = 1.0


@struct.dataclass
class RolloutState:
    action_trace: chex.Array
    episode_length: int


def get_action_counts(state: RolloutState, num_actions: int) -> chex.Array:
    valid_actions = state.action_trace[: state.episode_length].astype(jnp.int32)
    actions, counts = jnp.unique(valid_actions, return_counts=True)
    return jnp.zeros(num_actions, dtype=jnp.int32).at[actions].set(counts)


def _update_rollout_state(
    state: RolloutState,
    s_t: ContinuousTimeEnvState,
    a: ActT,
    r: float,
    s_tp1: ContinuousTimeEnvState,
) -> RolloutState:
    return dataclasses.replace(
        state,
        action_trace=state.action_trace.at[state.episode_length].set(a),
        episode_length=state.episode_length + (1 - s_t.done),
    )


RolloutCarry = tuple[ContinuousTimeEnvState, RolloutState]


class ContinuousTimeEnv(gym.Env[ObsT, ActT], abc.ABC):
    def __init__(self, timestep):
        self.rng = np.random.RandomState()
        self.timestep = timestep

    def seed(self, s):
        self.rng.seed(s)

    @abc.abstractmethod
    def reset_stateless(
        self, rng: chex.PRNGKey, train: bool = False
    ) -> ContinuousTimeEnvState: ...

    # @property
    # @abc.abstractmethod
    # def max_steps(self) -> int: ...

    @classmethod
    @abc.abstractmethod
    def step_stateless(
        rng: chex.PRNGKey, state: ContinuousTimeEnvState, action: ActT
    ) -> tuple[ContinuousTimeEnvState, float]: ...

    @classmethod
    @abc.abstractmethod
    def observation(state: ContinuousTimeEnvState) -> chex.Array: ...

    @classmethod
    def rollout_stateless(
        cls,
        rng: chex.PRNGKey,
        state: ContinuousTimeEnvState,
        policy: train_state.TrainState,
        num_steps: int,
        discount: float = 1.0,
    ) -> tuple[float, RolloutState]:
        def _env_step(c: RolloutCarry, i: int) -> tuple[RolloutCarry, float]:
            state_, rollout_state_ = c
            action_key, transition_key = jax.random.split(jax.random.fold_in(rng, i))
            action = policy.apply_fn(
                policy.params, cls.observation(state_), method="act"
            )
            next_state, reward = cls.step_stateless(transition_key, state_, action)
            reward = jax.lax.select(next_state.done, reward, next_state.h * reward)
            next_rollout_state = _update_rollout_state(
                rollout_state_, state_, action, reward, next_state
            )
            return (next_state, next_rollout_state), reward

        @jax.jit
        def step(carry: RolloutCarry, i: int) -> tuple[RolloutCarry, float]:
            state_, rollout_state_ = carry
            return jax.lax.cond(state_.done, lambda c, _: (c, 0.0), _env_step, carry, i)

        (final_state, rollout_stats), rewards = jax.lax.scan(
            step,
            (
                state,
                RolloutState(
                    action_trace=jnp.zeros((num_steps, *state.action_shape)),
                    episode_length=0,
                ),
            ),
            jnp.arange(num_steps),
        )
        discounts = discount ** (final_state.h * jnp.arange(num_steps))
        # TODO: Implement continuous-time returns
        # discounted_rewards = (
        #     (discounts * rewards).at[rollout_stats.episode_length :].set(0.0)
        # )
        # return jnp.sum(discounted_rewards), rollout_stats
        return rewards.dot(discounts), rollout_stats


@typing.runtime_checkable
class ContinuousTimeEnvSpec(Generic[ObsT, ActT], Protocol):
    def __call__(self, timestep: float) -> ContinuousTimeEnv[ObsT, ActT]: ...
