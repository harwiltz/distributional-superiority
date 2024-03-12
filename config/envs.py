import enum
import functools
from typing import Any, TypeVar

import fiddle as fdl
from fiddle.experimental import auto_config
from fiddle.selectors import select

from dsup.envs.continuous_time_env import ContinuousTimeEnvSpec
from dsup.envs.lim_malek import LimMalekOptionTrading
from dsup.tags import Environment
from dsup.trainer import Trainer


class EnvironmentName(enum.Enum):
    LIM_MALEK = "options"


@auto_config.auto_config
def lim_malek_spec(idx=0) -> ContinuousTimeEnvSpec[Any, Any]:
    return functools.partial(
        LimMalekOptionTrading, horizon=100.0, stock_idx=idx, K=1.0, s0=1.0
    )


def set_env_lim_malek(cfg: fdl.Buildable[Trainer[Any]], idx=0):
    cfg.env_spec = lim_malek_spec.as_buildable(idx=idx)
    cfg.num_steps = 120_000
    cfg.warmup_steps = 0
    cfg.discount = 0.999
    cfg.buffer.capacity = 20_000


def set_env(cfg: fdl.Buildable[Trainer[Any]], env_name: EnvironmentName | str):
    match env_name:
        case EnvironmentName.LIM_MALEK:
            set_env_lim_malek(cfg)
