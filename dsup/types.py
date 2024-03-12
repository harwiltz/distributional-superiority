import collections
from typing import (
    Hashable,
    Iterable,
    Mapping,
    TypeVar,
)

import chex
import gymnasium as gym

T = TypeVar("T")
ObsT = TypeVar("ObsT")
ActT = TypeVar("ActT")

PyTreeLeaf = TypeVar("PyTreeLeaf")
PyTree = (
    PyTreeLeaf
    | Iterable["PyTree[PyTreeLeaf]"]
    | Mapping[Hashable, "PyTree[PyTreeLeaf]"]
)

TransitionBatch = collections.namedtuple(
    "TransitionBatch", ["o_tm1", "a_tm1", "r_tm1", "m_t", "o_t"]
)
TransitionBatch.__doc__ = """
A batch of transition data, composed of:
    o_tm1 (state)  : observation at timestep t - 1
    a_tm1 (action) : action at timestep t - 1
    r_tm1 (float)  : reward at timestep t - 1
    m_t   (bool)   : mask at timestep t -- when False, don't bootstrap
    o_t   (state)  : observation at timestep t
"""

LossArtifacts = dict[str, chex.Scalar]
