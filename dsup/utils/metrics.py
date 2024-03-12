import logging
from typing import TypeVar

import flax
import jax
import jax.numpy as jnp
from clu import metrics as clu_metrics

M = TypeVar("M", bound=clu_metrics.Metric)


@flax.struct.dataclass
class ArrayAverage(clu_metrics.Metric):
    total: jnp.ndarray
    count: jnp.ndarray

    @classmethod
    def from_model_output(
        cls, value: jnp.ndarray, *, shape: jax.ShapeDtypeStruct, **kwargs
    ) -> clu_metrics.Metric:
        formatted = jnp.reshape(value, (-1, *shape.shape))
        return cls(
            total=jnp.sum(formatted, axis=0),
            count=formatted.shape[0],
        )

    @classmethod
    def from_output(cls, name: str, shape: jax.ShapeDtypeStruct, **kwargs):
        @flax.struct.dataclass
        class FromOutput(cls):
            """Wrapper Metric class that collects output named `name`."""

            @classmethod
            def from_model_output(cls: type[M], **model_output) -> M:
                output = jnp.array(model_output[name])
                mask = model_output.get("mask")
                if mask is not None and (output.shape or [0])[0] != mask.shape[0]:
                    logging.warning(
                        "Ignoring mask for model output '%s' because of shape mismatch: "
                        "output.shape=%s vs. mask.shape=%s",
                        name,
                        output.shape,
                        mask.shape,
                    )
                    mask = None
                return super().from_model_output(output, shape=shape, mask=mask)

            @classmethod
            def empty(cls: type[M]) -> M:
                return super().empty(shape=shape)

        return FromOutput

    def merge(self, other: clu_metrics.Metric) -> clu_metrics.Metric:
        return type(self)(
            total=self.total + other.total,
            count=self.count + other.count,
        )

    def compute(self):
        return self.total / self.count

    @classmethod
    def empty(cls, *, shape: jax.ShapeDtypeStruct):
        return cls(total=jnp.zeros(shape.shape, dtype=shape.dtype), count=0)
