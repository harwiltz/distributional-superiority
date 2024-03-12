import logging
from typing import Any, Mapping

import aim
from clu.metric_writers.interface import Array, MetricWriter


class AimLogHandler(logging.Handler):
    """Logging handler for aim.
    If user writes logs with the `logging` library, these will be tunneled
    to aim.
    """

    def __init__(self, run: aim.Run):
        super().__init__()
        self.run = run

    def emit(self, record: logging.LogRecord):
        match record.levelno:
            case logging.DEBUG:
                self.run.log_debug(record.getMessage())
            case logging.INFO:
                self.run.log_info(record.getMessage())
            case logging.WARNING:
                self.run.log_warning(record.getMessage())
            case logging.ERROR:
                self.run.log_error(record.getMessage())
            case _:
                ...


class AimWriter(MetricWriter):
    def __init__(self, **kwargs):
        self.run = aim.Run(**kwargs)
        logging.getLogger("aim").addHandler(AimLogHandler(self.run))

    def write_summaries(
        self,
        step: int,
        values: Mapping[str, Array],
        metadata: Mapping[str, Any] | None = None,
    ):
        for key, value in values.items():
            self.run.track(value, name=key, step=step, context=metadata)
        self.run.report_progress()

    def write_images(self, step: int, images: Mapping[str, Array]):
        for key, value in images.items():
            self.run.track(aim.Image(value), name=key, step=step)
        self.run.report_progress()

    def write_scalars(self, step: int, scalars: Mapping[str, float]):
        for key, value in scalars.items():
            self.run.track(float(value), name=key, step=step)
        self.run.report_progress()

    def write_hparams(self, hparams: Mapping[str, Any]):
        self.run["hparams"] = hparams

    def write_audios(self, *args, **kwargs):
        raise NotImplementedError

    def write_histograms(
        self,
        step: int,
        arrays: Mapping[str, Array],
        num_buckets: Mapping[str, int] | None = None,
    ):
        for key, array in arrays.items():
            dist = aim.Distribution(
                array, bin_count=num_buckets[key] if num_buckets else 64
            )
            self.run.track(dist, name=key, step=step)
        self.run.report_progress()

    def write_videos(self, *args, **kwargs):
        raise NotImplementedError

    def write_texts(self, step: int, texts: Mapping[str, str]):
        for key, text in texts.items():
            self.run.track(aim.Text(text), name=key, step=step)

    def close(self):
        self.run.close()

    def flush(self):
        pass
