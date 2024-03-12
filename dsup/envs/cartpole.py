import gymnasium as gym
import numpy as np
from gymnasium.envs.classic_control.cartpole import CartPoleEnv
from gymnasium.wrappers.time_limit import TimeLimit


class _ContinuousTimeCartPole(CartPoleEnv):
    def __init__(
        self,
        render_mode: str | None = None,
        timestep_multiplier: float = 1.0,
    ):
        super().__init__(render_mode=render_mode)
        self.tau = self.tau * timestep_multiplier

    @property
    def timestep(self):
        return self.tau

    def seed(self, seed):
        self.np_random = np.random.default_rng(seed)


def ContinuousTimeCartPole(
    render_mode: str | None = None,
    timestep_multiplier: float = 1.0,
    max_episode_steps: int = 500,
):
    return TimeLimit(
        _ContinuousTimeCartPole(
            render_mode=render_mode, timestep_multiplier=timestep_multiplier
        ),
        max_episode_steps,
    )
