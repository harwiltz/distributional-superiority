import functools
from typing import Annotated, TypeVar

import fiddle as fdl
import gymnasium as gym
import jax
import jax.numpy as jnp
import optax
from absl import flags
from fiddle.experimental import auto_config

from config import envs as env_configs
from dsup import statistical_functionals, train_state
from dsup.agents.qrdqn import QRDQNTrainer
from dsup.envs.cartpole import ContinuousTimeCartPole
from dsup.envs.lim_malek import LimMalekOptionTrading
from dsup.models import ActionConditionedHead, MLPTorso, NeuralNet
from dsup.replay import CircularReplayBuffer
from dsup.tags import Environment, NumAtoms, Timestep
from dsup.utils import schedules, stochastic_process

ObsT = TypeVar("ObsT")
ActT = TypeVar("ActT")

FLAGS = flags.FLAGS


@auto_config.auto_config
def base() -> QRDQNTrainer:
    num_atoms: Annotated[int, NumAtoms] = 100

    return QRDQNTrainer(
        env_spec=functools.partial(ContinuousTimeCartPole),
        num_atoms=num_atoms,
        buffer=CircularReplayBuffer(capacity=100_000),
        num_steps=10_000,
        write_metrics_interval_steps=1000,
        eval_interval_steps=10_000,
        seed=42,
        batch_size=8,
        quantile_model=MLPTorso(num_layers=2, num_hidden_units=100),
        optim=optax.adam(learning_rate=1e-4),
        target_params_update=train_state.HardTargetParamsUpdate(update_period=1000),
        discount=0.99,
        kappa=1.0,
        statistical_functional=statistical_functionals.MeanFunctional(),
        exploration_schedule=schedules.LinearSchedule(
            init_value=1.0, end_value=0.02, horizon=20_000
        ),
        warmup_steps=10_000,
        decision_frequency=1,
        eval_functional=statistical_functionals.MeanFunctional(),
    )


def lim_malek(idx: int = 0) -> fdl.Config[QRDQNTrainer]:
    cfg = base.as_buildable()
    env_configs.set_env_lim_malek(cfg, idx=idx)
    cfg.quantile_model.num_layers = 2
    cfg.quantile_model.num_hidden_units = 100
    cfg.num_atoms = 100
    cfg.optim.learning_rate = 1e-4
    cfg.batch_size = 32
    cfg.exploration_schedule.end_value = 0.02
    cfg.exploration_schedule.horizon = 20_000
    return cfg


def debug() -> fdl.Config[QRDQNTrainer]:
    cfg = base.as_buildable()
    cfg.num_atoms = 3
    cfg.batch_size = 64
    cfg.num_steps = 500_000
    cfg.target_params_update = train_state.HardTargetParamsUpdate(update_period=1_000)
    cfg.quantile_model.num_layers = 3
    cfg.quantile_model.num_hidden_units = 32
    cfg.optim.learning_rate = 6e-3
    cfg.eval_interval_steps = 1_000
    cfg.exploration_schedule = schedules.LinearSchedule(
        init_value=1.0, end_value=0.05, horizon=50_000
    )
    return cfg
