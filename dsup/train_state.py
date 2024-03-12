import dataclasses
import typing
from typing import Protocol

import jax
import optax
from clu import metrics as clu_metrics
from flax import struct
from flax.training import train_state
from typing_extensions import Self

from dsup.types import PyTree
from dsup.utils.stochastic_process import StochasticProcess, ZeroProcess


class TrainState(train_state.TrainState):
    variables: PyTree[jax.Array]
    metrics: clu_metrics.Collection = struct.field(
        default_factory=clu_metrics.Collection.create_collection
    )


@typing.runtime_checkable
class TargetParamsUpdate(Protocol):
    def __call__(
        self,
        *,
        old_params: PyTree[jax.Array],
        new_params: PyTree[jax.Array],
        steps: int,
    ) -> PyTree[jax.Array]: ...


@dataclasses.dataclass(frozen=True)
class HardTargetParamsUpdate:
    update_period: int

    def __call__(
        self,
        *,
        old_params: PyTree[jax.Array],
        new_params: PyTree[jax.Array],
        steps: int,
    ) -> PyTree[jax.Array]:
        return optax.periodic_update(new_params, old_params, steps, self.update_period)


@dataclasses.dataclass(frozen=True)
class SoftTargetParamsUpdate:
    step_size: float

    def __call__(
        self,
        *,
        old_params: PyTree[jax.Array],
        new_params: PyTree[jax.Array],
        steps: int,
    ) -> PyTree[jax.Array]:
        return optax.incremental_update(new_params, old_params, self.step_size)


class FittedValueTrainState(train_state.TrainState):
    target_params: PyTree[jax.Array]
    target_params_update: TargetParamsUpdate = struct.field(pytree_node=False)
    metrics: clu_metrics.Collection = struct.field(
        default_factory=clu_metrics.Collection.create_collection
    )

    def apply_gradients(
        self,
        /,
        grads: optax.Updates,
        metrics: clu_metrics.Collection | None = None,
        **kwargs,
    ) -> Self:
        state = super().apply_gradients(grads=grads)
        new_metrics = self.metrics.merge(metrics) if metrics else self.metrics
        new_target_params = self.target_params_update(
            new_params=state.params, old_params=state.target_params, steps=state.step
        )
        return state.replace(
            target_params=new_target_params, metrics=new_metrics, **kwargs
        )

    @classmethod
    def create(
        cls,
        /,
        *,
        params: PyTree[jax.Array],
        target_params_update: TargetParamsUpdate,
        **kwargs,
    ) -> Self:
        # need x.copy() to allow donation
        target_params = jax.tree_util.tree_map(lambda x: x.copy(), params)
        return super().create(
            params=params,
            target_params=target_params,
            target_params_update=target_params_update,
            **kwargs,
        )


# class ExploratoryTrainState(struct.PyTreeNode):
#     model_state: train_state.TrainState
#     noise_state: StochasticProcess[jax.Array]

#     @property
#     def step(self):
#         return self.model_state.step

#     @property
#     def metrics(self):
#         return self.model_state.metrics


class ExploratoryTrainState(FittedValueTrainState):
    noise_state: jax.Array | None = None
