import unittest
from typing import Any

import fiddle as fdl
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
from flax import struct

from config.agents import dqn as dqn_configs
from dsup.envs.continuous_time_env import (
    ContinuousTimeEnv,
    ContinuousTimeEnvState,
    RolloutState,
    get_action_counts,
)
from dsup.train_state import FittedValueTrainState
from dsup.trainer import Trainer

TrainerConfig = fdl.Buildable[Trainer[FittedValueTrainState]]


@struct.dataclass
class DummyPolicy:
    action: int
    params: int

    def apply_fn(self, params, x, **kwargs):
        return self.action


@struct.dataclass
class DummyEnvState(ContinuousTimeEnvState):
    time: float
    horizon: float
    inner_reward: float
    terminal_reward: float


class DummyEnv(ContinuousTimeEnv[Any, Any]):
    def __init__(self, *, horizon, inner_reward, terminal_reward, dt):
        self.horizon = horizon
        self.inner_reward = inner_reward
        self.terminal_reward = terminal_reward
        self.dt = dt
        self.observation_space = gym.spaces.Box(np.zeros(2), np.ones(2))
        self.action_space = gym.spaces.Discrete(1)

    def seed(self, _): ...

    def reset_stateless(self, _):
        return DummyEnvState(
            time=0.0,
            horizon=self.horizon,
            action_shape=(1,),
            done=False,
            h=self.dt,
            inner_reward=self.inner_reward,
            terminal_reward=self.terminal_reward,
        )

    @classmethod
    def step_stateless(cls, rng, state, action) -> tuple[DummyEnvState, float]:
        next_time = state.time + state.h
        next_state = state.replace(time=next_time, done=next_time >= state.horizon)
        return next_state, next_state.done * state.terminal_reward + (
            1 - next_state.done
        ) * state.inner_reward

    @classmethod
    def observation(cls, state):
        return jnp.zeros(2)


class TestStatelessEnv(unittest.TestCase):
    def setUp(self):
        self.configs = [
            dqn_configs.lim_malek,
        ]
        self.rng = jax.random.PRNGKey(0)
        self.num_envs = 5

    def test_stateless_rollout(self):
        for config in self.configs:
            with self.subTest(msg=f"Single rollout ({config.__name__})", config=config):
                trainer = fdl.build(config())
                initial_state = trainer.env.reset_stateless(self.rng)
                score, data = trainer.env.rollout_stateless(
                    self.rng, initial_state, trainer.state, trainer.env.max_steps
                )
                self.assertIsInstance(score.item(), float)
                self.assertListEqual(
                    list(data.action_trace.shape),
                    list((trainer.env.max_steps, *trainer.env.action_space.shape)),
                )

    def test_vectorized_stateless_reset(self):
        for config in self.configs:
            with self.subTest(
                msg=f"Vectorized reset ({config.__name__})", config=config
            ):
                trainer = fdl.build(config())
                init_rngs = jax.random.split(self.rng, self.num_envs)
                initial_states = jax.vmap(trainer.env.reset_stateless)(init_rngs)
                count = jax.eval_shape(lambda x: x.done, initial_states).shape[0]
                self.assertEqual(count, self.num_envs)

    def test_vectorized_stateless_rollout(self):
        for config in self.configs:
            with self.subTest(
                msg=f"Vectorized rollout ({config.__name__})", config=config
            ):
                trainer = fdl.build(config())
                init_rng, rollout_rng = jax.random.split(self.rng)
                init_rngs = jax.random.split(init_rng, self.num_envs)
                initial_states = jax.vmap(trainer.env.reset_stateless)(init_rngs)
                rollout_rngs = jax.random.split(rollout_rng, self.num_envs)
                returns, data = jax.vmap(
                    trainer.env.rollout_stateless, in_axes=(0, 0, None, None)
                )(rollout_rngs, initial_states, trainer.state, trainer.env.max_steps)
                self.assertListEqual(list(returns.shape), [self.num_envs])

    def test_action_counts(self):
        num_zeros = 5
        num_ones = 2
        real_counts = [num_zeros, num_ones]
        episode_length = num_zeros + num_ones
        actions_valid = jnp.array(num_zeros * [0] + num_ones * [1])
        action_trace = (
            (2 * jnp.ones(2 * episode_length)).at[:episode_length].set(actions_valid)
        )
        state = RolloutState(action_trace=action_trace, episode_length=episode_length)
        counts = get_action_counts(state, 3)
        self.assertListEqual(list(counts), real_counts + [0])

    @unittest.skip("Cannot vectorize action counts yet")
    def test_action_counts_vectorized(self):
        episode_length = 100
        cutoff = 30
        action_valid_1 = jnp.array(cutoff * [0] + (episode_length - cutoff) * [1])
        action_valid_2 = jnp.array(cutoff * [1] + (episode_length - cutoff) * [0])
        action_trace_1 = (
            (2 * jnp.ones(2 * episode_length)).at[:episode_length].set(action_valid_1)
        )
        action_trace_2 = (
            (2 * jnp.ones(2 * episode_length)).at[:episode_length].set(action_valid_2)
        )
        action_traces = jnp.stack([action_trace_1, action_trace_2])
        states = jax.vmap(RolloutState, in_axes=(0, None))(
            action_traces, episode_length
        )
        counts = jax.vmap(get_action_counts, in_axes=(0, None))(states, 3)
        self.assertListEqual(list(counts.shape), [2, 3])
        self.assertListEqual(list(counts.shape), [2, 3])

    def test_terminal_returns(self):
        horizon = 5
        dt = 0.1
        inner_reward = 0
        terminal_reward = 1.0
        env = DummyEnv(
            horizon=horizon,
            inner_reward=inner_reward,
            terminal_reward=terminal_reward,
            dt=dt,
        )
        init_state = env.reset_stateless(None)
        result = env.rollout_stateless(
            jax.random.PRNGKey(0),
            init_state,
            DummyPolicy(0, 0),
            int(2 * horizon / dt),
            discount=1.0,
        )
        ret, _ = result
        self.assertAlmostEqual(ret, terminal_reward, places=5)

    def test_inner_returns(self):
        horizon = 5
        dt = 0.1
        inner_reward = 1.0
        terminal_reward = 0.0
        env = DummyEnv(
            horizon=horizon,
            inner_reward=inner_reward,
            terminal_reward=terminal_reward,
            dt=dt,
        )
        init_state = env.reset_stateless(None)
        result = env.rollout_stateless(
            jax.random.PRNGKey(0),
            init_state,
            DummyPolicy(0, 0),
            int(2 * horizon / dt),
            discount=1.0,
        )
        ret, _ = result
        self.assertAlmostEqual(ret, inner_reward * horizon, places=5)


if __name__ == "__main__":
    unittest.main()
