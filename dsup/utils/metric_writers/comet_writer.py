from typing import Any, Mapping

import comet_ml
import matplotlib
import matplotlib.figure
from chex import Array
from clu import metric_writers


class CometWriter(metric_writers.MetricWriter):
    def __init__(self, exp_name: str):
        comet_ml.init(project_name="distributional-superiority")
        self.exp: comet_ml.Experiment = comet_ml.Experiment(
            project_name="distributional-superiority",
        )
        self.exp.add_tag(exp_name)

    def write_summaries(
        self,
        step: int,
        values: Mapping[str, Array],
        metadata: Mapping[str, Any] | None = None,
    ):
        for key, value in values.items():
            self.exp.log_other({key: value, **(metadata or {})}, step=step)

    def write_images(self, step: int, images: Mapping[str, Array]):
        for key, value in images.items():
            if isinstance(value, matplotlib.figure.Figure):
                self.exp.log_figure(
                    figure=value, figure_name=key, step=step, overwrite=True
                )
            else:
                self.exp.log_image(value, name=key, overwrite=True, step=step)

    def write_scalars(self, step: int, scalars: Mapping[str, float]):
        self.exp.log_metrics(scalars, step=step)

    def write_hparams(self, hparams: Mapping[str, Any]):
        self.exp.log_parameters(hparams)

    def write_audios(self, step: int, audios: Mapping[str, Array], *, sample_rate: int):
        raise NotImplementedError

    def write_histograms(
        self,
        step: int,
        arrays: Mapping[str, Array],
        num_buckets: Mapping[str, int] | None = None,
    ):
        raise NotImplementedError

    def write_videos(self, step: int, videos: Mapping[str, Array]):
        raise NotImplementedError

    def write_texts(self, step: int, texts: Mapping[str, str]):
        pass

    def close(self):
        pass

    def flush(self):
        pass
