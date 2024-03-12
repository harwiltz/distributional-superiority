from typing import Any

import fiddle as fdl

from dsup import statistical_functionals
from dsup.statistical_functionals import SampleStatisticalFunctional
from dsup.trainer import Trainer


def set_decision_frequency(
    base: fdl.Buildable[Trainer[Any]], decision_frequency: int = 1
):
    base.decision_frequency = decision_frequency
    # base.num_steps = base.num_steps * decision_frequency


def dsup_proper(base: fdl.Buildable[Trainer[Any]]):
    if hasattr(base, "rescale_factor"):
        base.rescale_factor = 0.5
        base.shift_by_q = False


def dsup_shifted(base: fdl.Buildable[Trainer[Any]], rescale_factor: float = 0.5):
    if hasattr(base, "shift_by_q"):
        base.shift_by_q = True
        base.rescale_factor = rescale_factor


def set_decision_period(base: fdl.Buildable[Trainer[Any]], h: float = 1.0):
    base.h = h
    base.env.timestep = h
    base.buffer.capacity = int(base.buffer.capacity / h)
    base.optim.learning_rate *= h
    base.target_params_update.update_period = int(
        base.target_params_update.update_period / h
    )
    base.warmup_steps = int(base.warmup_steps / h)
    base.num_steps = int(base.num_steps / h)


def set_risk_measure_cvar(base: fdl.Buildable[Trainer[Any]], alpha: float, both=True):
    base.eval_functional = fdl.Config(
        statistical_functionals.CVaRFunctional, alpha=alpha, requires_sort=True
    )
    if both and hasattr(base, "statistical_functional"):
        base.statistical_functional = fdl.Config(
            statistical_functionals.CVaRFunctional, alpha=alpha, requires_sort=True
        )
