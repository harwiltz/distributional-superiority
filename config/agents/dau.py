import copy
import functools
from typing import Annotated

import fiddle as fdl
import optax
from fiddle.experimental import auto_config

from config import envs as env_configs
from dsup import train_state
from dsup.agents.dau import DAUTrainer
from dsup.envs.cartpole import ContinuousTimeCartPole
from dsup.models import MLPTorso
from dsup.replay import CircularReplayBuffer
from dsup.tags import Timestep
from dsup.utils import schedules, stochastic_process


@auto_config.auto_config
def base() -> DAUTrainer:
    return DAUTrainer(
        env_spec=functools.partial(ContinuousTimeCartPole),
        buffer=CircularReplayBuffer(capacity=100_000),
        num_steps=200_000,
        write_metrics_interval_steps=1_000,
        eval_interval_steps=10_000,
        seed=42,
        batch_size=64,
        advantage_model=MLPTorso(num_layers=3, num_hidden_units=128),
        value_model=MLPTorso(num_layers=3, num_hidden_units=128),
        optim=optax.adam(learning_rate=6e-3),
        target_params_update=train_state.HardTargetParamsUpdate(update_period=1000),
        discount=0.999,
        exploration_process=stochastic_process.OrnsteinUhlenbeckProcess(
            stiffness=0.75, scale=0.5
        ),
        exploration_schedule=schedules.ConstantSchedule(value=1.0),
        warmup_steps=10_000,
        decision_frequency=1,
    )


def lim_malek(
    timestep_multiplier: float = 1.0,
    idx: int = 0,
) -> fdl.Config[DAUTrainer]:
    cfg = base.as_buildable()
    env_configs.set_env_lim_malek(cfg, idx=idx)
    cfg.advantage_model.num_layers = 2
    cfg.advantage_model.num_hidden_units = 100
    cfg.optim.learning_rate = 1e-4
    cfg.batch_size = 32
    return cfg


def debug() -> fdl.Config[DAUTrainer]:
    cfg = base.as_buildable()
    cfg.advantage_model.num_hidden_units = 16
    cfg.value_model = copy.deepcopy(cfg.advantage_model)
    cfg.target_params_update = train_state.HardTargetParamsUpdate(update_period=1000)
    cfg.batch_size = 64
    return cfg
