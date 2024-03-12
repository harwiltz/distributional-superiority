import functools
from typing import Annotated

import fiddle as fdl
import gymnasium as gym
import optax
from fiddle.experimental import auto_config

import dsup.models as models
import dsup.tags as tags
from config import envs as env_configs
from dsup import statistical_functionals, train_state
from dsup.agents.dsup import DistributionalSuperiorityTrainer
from dsup.envs.cartpole import ContinuousTimeCartPole
from dsup.envs.lim_malek import LimMalekOptionTrading
from dsup.replay import CircularReplayBuffer
from dsup.utils import schedules, stochastic_process

DiscreteActionEnv = gym.Env[gym.spaces.Box, gym.spaces.Discrete]


@auto_config.auto_config
def _base() -> DistributionalSuperiorityTrainer:
    capacity: Annotated[int, tags.ReplayCapacity] = 10_000
    num_atoms: Annotated[int, tags.NumAtoms] = 100
    num_hidden_units: int = 100

    return DistributionalSuperiorityTrainer(
        env_spec=functools.partial(ContinuousTimeCartPole),
        little_q_model=models.MLPTorso(num_layers=3, num_hidden_units=num_hidden_units),
        superiority_model=models.MLPTorso(
            num_layers=3, num_hidden_units=num_hidden_units
        ),
        eta_model=models.MLPTorso(num_layers=2, num_hidden_units=num_hidden_units),
        num_atoms=num_atoms,
        num_steps=500_000,
        discount=0.99,
        kappa=1.0,
        buffer=CircularReplayBuffer(capacity=capacity),
        write_metrics_interval_steps=1000,
        eval_interval_steps=10_000,
        seed=42,
        batch_size=64,
        optim=optax.adam(
            learning_rate=6e-3,
        ),
        target_params_update=train_state.HardTargetParamsUpdate(update_period=1000),
        statistical_functional=statistical_functionals.MeanFunctional(),
        # exploration_process=stochastic_process.OrnsteinUhlenbeckProcess(
        #     stiffness=0.75, scale=1.5
        # ),
        exploration_process=stochastic_process.ZeroProcess(),
        exploration_schedule=schedules.LinearSchedule(
            init_value=1.0, end_value=0.02, horizon=20_000
        ),
        warmup_steps=0,
        rescale_factor=0.5,
        shift_by_q=False,
        decision_frequency=1,
        eval_functional=statistical_functionals.MeanFunctional(),
    )


def base() -> fdl.Config[DistributionalSuperiorityTrainer]:
    cfg = _base.as_buildable()
    cfg.superiority_model = cfg.little_q_model
    return cfg


def lim_malek(
    timestep_multiplier: float = 1.0,
    idx: int = 0,
) -> fdl.Config[DistributionalSuperiorityTrainer]:
    cfg = base()
    env_configs.set_env_lim_malek(cfg, idx=idx)
    cfg.little_q_model.num_layers = 2
    cfg.little_q_model.num_hidden_units = 100
    cfg.superiority_model = cfg.little_q_model
    cfg.eta_model.num_layers = 2
    cfg.eta_model.num_hidden_units = 100
    cfg.optim.learning_rate = 1e-4
    cfg.batch_size = 32
    return cfg


def debug() -> fdl.Config[DistributionalSuperiorityTrainer]:
    cfg = base()
    cfg.num_atoms = 3
    cfg.superiority_model.num_hidden_units = 32
    cfg.little_q_model = cfg.superiority_model
    cfg.exploration_process.scale = 0.5
    return cfg
