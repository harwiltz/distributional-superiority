import collections
from typing import Any

import chex
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
from etils import epath
from flax import struct

from dsup.envs.continuous_time_env import ContinuousTimeEnv, ContinuousTimeEnvState

BASE_TIMESTEP = 1.0

LimMalekParams = collections.namedtuple(
    "LimMalekParams", ["drift", "volatility", "K", "s0", "horizon", "dt"]
)


@struct.dataclass
class LimMalekState(ContinuousTimeEnvState):
    time: float
    value: float
    params: LimMalekParams


VAL_CUTOFF = 1962  # Standard cut-off for train/val split


class LimMalekOptionTrading(ContinuousTimeEnv[gym.spaces.Box, gym.spaces.Discrete]):
    def __init__(
        self,
        timestep,
        data_path: epath.Path = epath.Path("data/dow_top10_2005_2019.npy"),
        horizon: float = 100.0,
        stock_idx: int | None = 0,
        K: float = 1.0,
        s0: float = 1.0,
        test: bool = False,
    ):
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(
            -np.inf, np.inf, shape=(2,), dtype="float64"
        )
        data = np.load(data_path.resolve())
        self.stock_idx = stock_idx
        if stock_idx is not None:
            data = data[stock_idx, :, 3]  # index 3 holds stock prices
        if test:
            data = data[: VAL_CUTOFF - 1]
        else:
            data = data[VAL_CUTOFF:-1]
        data_log_ratio = np.log(data[1:] / data[:-1])
        self.data_path = data_path
        self.drift = np.mean(data_log_ratio)
        self.volatility = np.std(data_log_ratio)
        self.horizon = horizon
        self.K = K
        self.s0 = s0
        self.state_t = 0.0
        self.state_x = s0
        self.h = timestep
        self.timestep = timestep * BASE_TIMESTEP
        self.params = LimMalekParams(
            drift=self.drift,
            volatility=self.volatility,
            s0=self.s0,
            K=self.K,
            horizon=self.horizon,
            dt=self.timestep,
        )
        self.max_steps = int(np.ceil(self.horizon / self.timestep))
        super().__init__(timestep)

    def test(self):
        return LimMalekOptionTrading(
            self.h,
            self.data_path,
            horizon=self.horizon,
            stock_idx=self.stock_idx,
            K=self.K,
            s0=self.s0,
            test=True,
        )

    def obs(self):
        return np.array([self.state_x, (self.horizon - self.state_t) / self.horizon])

    def step(
        self, action: int
    ) -> tuple[np.ndarray[float], float, bool, bool, dict[str, Any]]:
        if self.state_t >= self.horizon - self.timestep:
            action = 1
        self.state_t += self.timestep
        if action > 0:
            r = np.maximum(0, self.K - self.state_x)
            done = True
        else:
            r = 0.0
            done = False
        h = self.timestep * BASE_TIMESTEP
        log_dXt = self.drift * h + self.volatility * np.sqrt(h) * self.rng.randn()
        self.state_x = self.state_x * np.exp(log_dXt)
        self.state_t += h
        return self.obs(), r, done, False, {}

    def reset(self, train: bool = True):
        max_init_t = self.horizon * 0.98
        if train:
            self.state_t = self.rng.uniform(0.0, max_init_t)
        else:
            self.state_t = 0.0
        self.state_x = self.s0
        return self.obs(), {}

    def reset_stateless(self, rng: chex.PRNGKey, train: bool = False) -> LimMalekState:
        max_init_t = train * (self.horizon * 0.98)
        start_time = jax.random.uniform(rng, minval=0.0, maxval=max_init_t)
        return LimMalekState(
            action_shape=self.action_space.shape,
            time=start_time,
            value=self.s0,
            params=self.params,
            h=self.h,
        )

    @classmethod
    def observation(cls, state: LimMalekState) -> chex.Array:
        return jnp.array(
            [state.value, (state.params.horizon - state.time) / state.params.horizon]
        )

    @classmethod
    def step_stateless(
        cls, rng: chex.PRNGKey, state: LimMalekState, action: int
    ) -> tuple[LimMalekState, float]:
        dt = state.params.dt
        next_time = state.time + dt
        action = jax.lax.select(next_time >= state.params.horizon, 1, action)
        done = action > 0
        reward = jax.lax.select(
            action > 0, jnp.maximum(0, state.params.K - state.value), 0.0
        )
        noise = jax.random.normal(rng)
        log_dXt = (
            state.params.drift * dt + state.params.volatility * jnp.sqrt(dt) * noise
        )
        next_value = state.value * jnp.exp(log_dXt)
        return state.replace(time=next_time, value=next_value, done=done), reward
