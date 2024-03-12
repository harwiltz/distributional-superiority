import functools
from typing import Annotated, TypeVar

import fiddle as fdl
import optax
from absl import flags
from fiddle.experimental import auto_config

from config import envs as env_configs
from dsup import train_state
from dsup.agents.dqn import DQNTrainer
from dsup.envs.cartpole import ContinuousTimeCartPole
from dsup.envs.lim_malek import LimMalekOptionTrading
from dsup.models import MLPTorso
from dsup.replay import CircularReplayBuffer
from dsup.tags import Timestep
from dsup.utils import schedules, stochastic_process

ObsT = TypeVar("ObsT")
ActT = TypeVar("ActT")

FLAGS = flags.FLAGS


@auto_config.auto_config
def base() -> DQNTrainer:
    timestep: Annotated[float, Timestep] = 1.0

    return DQNTrainer(
        env_spec=functools.partial(ContinuousTimeCartPole),
        buffer=CircularReplayBuffer(capacity=100_000),
        num_steps=500_000,
        warmup_steps=1_000,
        write_metrics_interval_steps=1000,
        eval_interval_steps=10_000,
        seed=42,
        batch_size=64,
        q_model=MLPTorso(num_layers=3, num_hidden_units=16),
        optim=optax.adam(learning_rate=6e-3),
        target_params_update=train_state.HardTargetParamsUpdate(1000),
        discount=0.99,
        exploration_schedule=schedules.LinearSchedule(
            init_value=1.0, end_value=0.05, horizon=50_000
        ),
        decision_frequency=1,
    )


def lim_malek(timestep_multiplier: float = 1.0) -> fdl.Config[DQNTrainer]:
    cfg = base.as_buildable()
    env_configs.set_env_lim_malek(cfg)
    cfg.q_model.num_layers = 2
    cfg.q_model.num_hidden_units = 100
    cfg.optim.learning_rate = 1e-4
    cfg.batch_size = 32
    cfg.exploration_schedule.end_value = 0.02
    cfg.exploration_schedule.horizon = 20_000
    return cfg


def debug() -> fdl.Config[DQNTrainer]:
    cfg = base.as_buildable()
    cfg.eval_interval_steps = 1000
    cfg.num_steps = 500_000
    cfg.batch_size = 64
    cfg.target_params_update = train_state.HardTargetParamsUpdate(update_period=1000)
    cfg.q_model.num_layers = 3
    cfg.q_model.num_hidden_units = 16
    cfg.exploration_schedule = schedules.LinearSchedule(
        init_value=1.0, end_value=0.05, horizon=50_000
    )
    cfg.warmup_steps = 10_000
    # cfg.warmup_steps = 1
    return cfg
