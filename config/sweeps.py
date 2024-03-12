import copy
import functools
from typing import Any, Iterator

import fiddle as fdl
import numpy as np

from config import fiddlers
from dsup.trainer import Trainer


def sweep_decision_frequency(
    base: fdl.Buildable[Trainer[Any]], hs=[1, 5, 10, 15]
) -> Iterator[fdl.Buildable[Trainer[Any]]]:
    for h in list(hs):
        new_config = copy.deepcopy(base)
        fiddlers.set_decision_frequency(new_config, h)
        yield new_config


sweep_decision_frequency_all = functools.partial(
    sweep_decision_frequency, hs=[1, 5, 10, 15, 20, 25, 30, 35]
)
sweep_decision_frequency_short = functools.partial(
    sweep_decision_frequency, hs=[1, 5, 10, 15]
)
sweep_decision_frequency_long = functools.partial(
    sweep_decision_frequency, hs=[20, 25, 30, 35]
)


def sweep_seed(
    base: fdl.Buildable[Trainer[Any]], num_seeds=10
) -> Iterator[fdl.Buildable[Trainer[Any]]]:
    for seed in range(num_seeds):
        yield fdl.deepcopy_with(base, seed=seed)


def sweep_dsup_naivety(
    base: fdl.Buildable[Trainer[Any]],
) -> Iterator[fdl.Buildable[Trainer[Any]]]:
    if not hasattr(base, "rescale_factor"):
        yield base
        return
    yield fdl.deepcopy_with(base, rescale_factor=0.5, shift_by_q=False)
    yield fdl.deepcopy_with(base, rescale_factor=0.5, shift_by_q=True)
    yield fdl.deepcopy_with(base, rescale_factor=1.0, shift_by_q=True)
    yield fdl.deepcopy_with(base, rescale_factor=1.0, shift_by_q=False)


def sweep_cvar(
    base: fdl.Buildable[Trainer[Any]],
) -> Iterator[fdl.Buildable[Trainer[Any]]]:
    for alpha in np.linspace(1e-2, 1.0, 10):
        new_config = copy.deepcopy(base)
        fiddlers.set_risk_measure_cvar(base, float(alpha))
        yield new_config
