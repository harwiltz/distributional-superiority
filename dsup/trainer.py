import abc
import contextlib
import copy
import dataclasses
import functools
import logging
import os
import typing
from typing import Annotated, Generic, TypeVar

import chex
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as orbax
import tqdm
from absl import flags
from clu import metric_writers as clu_metric_writers
from clu import metrics as clu_metrics
from clu import periodic_actions
from etils import epath
from flax.training import train_state
from typing_extensions import Self

from dsup import statistical_functionals
from dsup.envs.continuous_time_env import (
    ContinuousTimeEnv,
    ContinuousTimeEnvSpec,
    get_action_counts,
)
from dsup.replay import CircularReplayBuffer
from dsup.tags import Environment
from dsup.types import TransitionBatch
from dsup.utils import metrics as dsup_metrics

StateT = TypeVar("StateT", bound=train_state.TrainState)
ObsT = TypeVar("ObsT")
ActT = TypeVar("ActT")

_WORKDIR = epath.DEFINE_path(name="workdir", default=None, help="Working directory")
_METRIC_WRITERS = flags.DEFINE_multi_enum(
    "metric_writer",
    "cli",
    ["aim", "wandb", "comet", "cli"],
    "Which metric writer to use",
)
_EXPERIMENT = flags.DEFINE_string(
    "experiment", default="dsup", help="Experiment name for tracking"
)
_NUM_EVAL_EPISODES = flags.DEFINE_integer(
    "num_eval_episodes",
    default=5,
    help="Number of episodes to aggregate for evaluation",
)
_ENABLE_STATELESS_EVAL = flags.DEFINE_boolean(
    "enable_stateless_eval",
    default=False,
    help="Enable vectorized stateless policy evaluation",
)
_ENABLE_CHECKPOINTS = flags.DEFINE_boolean(
    "enable_checkpoints", default=False, help="Enable model checkpointing"
)
_OVERWRITE_CHECKPOINTS = flags.DEFINE_boolean(
    "overwrite_checkpoints",
    default=False,
    help="Whether to overwrite model checkpoints",
)
_FINAL_NUM_EVAL_EPISODES = flags.DEFINE_integer(
    "final_num_eval_episodes",
    default=1000,
    help="Number of episodes to aggregate for last evaluation",
)

EVAL_MEAN_RETURN = "eval__mean_return"
EVAL_ACTION_COUNTS = "eval__action_counts"
EVAL_EPISODE_LENGTH = "eval__episode_length"
EVAL_SCALAR_TAGS = set([EVAL_MEAN_RETURN, EVAL_EPISODE_LENGTH])
EVAL_TEXT_TAGS = set([EVAL_ACTION_COUNTS])
EVAL_ARRAY_TAGS = set()


def make_final_key(key: str, delim="__"):
    prefix, suffix = key.split(delim, maxsplit=1)
    return delim.join(["final", suffix])


@dataclasses.dataclass(frozen=True, kw_only=True)
class Trainer(Generic[StateT], abc.ABC):
    env_spec: Annotated[ContinuousTimeEnvSpec[ObsT, ActT], Environment]
    buffer: CircularReplayBuffer
    num_steps: int
    warmup_steps: int
    write_metrics_interval_steps: int
    eval_interval_steps: int
    checkpoint_options: orbax.CheckpointManagerOptions = dataclasses.field(
        default_factory=orbax.CheckpointManagerOptions
    )
    seed: int
    batch_size: int
    decision_frequency: int  # 1 / h
    eval_functional: statistical_functionals.SampleStatisticalFunctional = (
        statistical_functionals.MeanFunctional()
    )

    @property
    def identifier(self) -> str:
        return self.__class__.__name__

    @property
    @abc.abstractmethod
    def state(self) -> StateT: ...

    def maybe_restore_state(self, state: StateT) -> StateT:
        if _OVERWRITE_CHECKPOINTS.value:
            return state

        if latest_step := self.checkpoint_manager.latest_step():
            logging.info(
                f"Restoring checkpoint from {_WORKDIR.value} at step {latest_step}."
            )
            restored = self.checkpoint_manager.restore(latest_step, state)
            return typing.cast(StateT, restored)

        logging.info("No checkpoint found")
        return state

    def maybe_save_state(
        self, step: int, state: StateT, *, force: bool = False
    ) -> None:
        if not _ENABLE_CHECKPOINTS.value:
            return

        if not force and not self.checkpoint_manager.should_save(step):
            return

        self.checkpoint_manager.save(
            step, state, metrics=state.metrics.compute(), force=force
        )

    @functools.cached_property
    def env(self) -> ContinuousTimeEnv[ObsT, ActT]:
        return self.env_spec(1.0 / self.decision_frequency)

    @functools.cached_property
    def checkpoint_manager(self) -> orbax.CheckpointManager:
        return orbax.CheckpointManager(
            directory=_WORKDIR.value.resolve(),
            checkpointers=orbax.StandardCheckpointer(),
            options=self.checkpoint_options,
        )

    @functools.cached_property
    def metric_writer(self) -> clu_metric_writers.MetricWriter:
        writer_enums = _METRIC_WRITERS.value
        if len(writer_enums) == 0:
            writer_enums = ["cli"]

        writers: list[clu_metric_writers.MetricWriter] = []

        for writer in _METRIC_WRITERS.value:
            match writer:
                case "cli":
                    writers.append(clu_metric_writers.LoggingWriter())
                case "aim":
                    from dsup.utils.metric_writers.aim_writer import AimWriter

                    writers.append(
                        AimWriter(experiment=_EXPERIMENT.value, log_system_params=True)
                    )
                case "wandb":
                    from dsup.utils.metric_writers.wandb_writer import WandBWriter

                    writers.append(
                        WandBWriter(
                            save_code=False, group=_EXPERIMENT.value, mode="online"
                        )
                    )
                case "comet":
                    from dsup.utils.metric_writers.comet_writer import CometWriter

                    writers.append(CometWriter(exp_name=_EXPERIMENT.value))
                case "tensorboard":
                    raise NotImplementedError(
                        f"Metric writer {writer} not yet supported."
                    )
                case _:
                    raise ValueError(f"Unknown metric writer: {writer}")
        return clu_metric_writers.MultiWriter(writers)

    @functools.cached_property
    def num_actions(self) -> int:
        return self.env.action_space.n

    @functools.cached_property
    def eval_metrics(self) -> clu_metrics.Collection:
        metric_manifest = {
            EVAL_MEAN_RETURN: clu_metrics.Average.from_output(EVAL_MEAN_RETURN),
            EVAL_EPISODE_LENGTH: clu_metrics.Average.from_output(EVAL_EPISODE_LENGTH),
            EVAL_ACTION_COUNTS: dsup_metrics.ArrayAverage.from_output(
                EVAL_ACTION_COUNTS,
                jax.ShapeDtypeStruct((self.num_actions,), jnp.int32),
            ),
        }
        return clu_metrics.Collection.create(**metric_manifest)

    @abc.abstractmethod
    def train_step(
        self, rng: chex.PRNGKey, state: StateT, batch: TransitionBatch, **kwargs
    ) -> StateT: ...

    @abc.abstractmethod
    def action(
        self,
        rng: chex.PRNGKey,
        state: StateT,
        obs: ObsT,
        *,
        train: bool,
        **kwargs,
    ) -> ActT: ...

    def eval(self, rng: chex.PRNGKey, state: StateT, num_episodes: int) -> None:
        metrics = self.eval_metrics.empty()
        env = self.env if not hasattr(self.env, "test") else self.env.test()
        if _ENABLE_STATELESS_EVAL.value and hasattr(self.env, "rollout_stateless"):
            init_rng, rollout_rng = jax.random.split(rng)
            init_rngs = jax.random.split(init_rng, num_episodes)
            initial_states = jax.vmap(self.env.reset_stateless)(init_rngs)
            rollout_rngs = jax.random.split(rollout_rng, num_episodes)
            returns, data = jax.vmap(
                self.env.rollout_stateless, in_axes=(0, 0, None, None)
            )(rollout_rngs, initial_states, state, self.env.max_steps)
            episode_length = data.episode_length
            action_count_list = [
                get_action_counts(
                    jax.tree_util.tree_map(lambda x: x[i], data),
                    self.env.action_space.n,
                )
                for i in range(_NUM_EVAL_EPISODES.value)
            ]
            action_counts = jnp.stack(action_count_list)
            episode_metrics = {
                EVAL_MEAN_RETURN: jnp.mean(returns).item(),
                EVAL_ACTION_COUNTS: action_counts,
                EVAL_EPISODE_LENGTH: episode_length,
            }
            metrics = metrics.merge(
                self.eval_metrics.single_from_model_output(**episode_metrics)
            )
        else:
            all_metrics = {
                EVAL_MEAN_RETURN: [],
                EVAL_ACTION_COUNTS: [],
                EVAL_EPISODE_LENGTH: [],
            }
            for i in range(num_episodes):
                key = jax.random.fold_in(rng, i)
                episode_metrics = self.run_episode(key, state)
                for key in all_metrics.keys():
                    all_metrics[key].append(episode_metrics[key])
            return_statistic = self.eval_functional.evaluate(
                jnp.array(all_metrics[EVAL_MEAN_RETURN])
            )
            all_metrics[EVAL_MEAN_RETURN] = return_statistic
            all_metrics[EVAL_ACTION_COUNTS] = jnp.concatenate(
                jnp.array(all_metrics[EVAL_ACTION_COUNTS]), axis=0
            )
            all_metrics[EVAL_EPISODE_LENGTH] = jnp.array(
                all_metrics[EVAL_EPISODE_LENGTH]
            )
            metrics = self.eval_metrics.single_from_model_output(**all_metrics)
        return metrics.compute()

    def run_episode(self, rng: chex.PRNGKey, state: StateT) -> None:
        env = copy.deepcopy(self.env)
        env.seed(state.step + rng[0])
        o_tm1, _ = env.reset(train=False)
        done = False
        score = 0.0
        h = 1.0 / self.decision_frequency
        action_counts = [0 for _ in range(env.action_space.n)]
        length = 0

        while not done:
            length += 1
            rng, a_rng = jax.random.split(rng)
            a_tm1 = np.asarray(self.action(a_rng, state, o_tm1, train=False)).item()
            action_counts[a_tm1] += 1
            o_t, r_tm1, d_t, t_t, _ = env.step(a_tm1)
            score += h * r_tm1 * (1 - d_t) + d_t * r_tm1
            done = t_t or d_t
            o_tm1 = o_t
        return {
            EVAL_MEAN_RETURN: score,
            EVAL_EPISODE_LENGTH: length,
            EVAL_ACTION_COUNTS: action_counts,
        }

    def train(self) -> None:
        np.random.seed(self.seed)
        self.env.seed(self.seed + 1)
        train_env = copy.deepcopy(self.env)
        train_env.seed(self.seed + 2)

        rng = jax.random.PRNGKey(self.seed)

        state = self.maybe_restore_state(self.state)

        num_env_steps = self.num_steps * self.decision_frequency

        report_progress = periodic_actions.ReportProgress(
            num_train_steps=self.num_steps, writer=self.metric_writer
        )

        def _write_metrics(step: int, t: float | None = None) -> None:
            nonlocal state
            del t
            self.metric_writer.write_scalars(step, state.metrics.compute())
            state = state.replace(metrics=state.metrics.empty())

        callbacks = [
            # lambda step: self.maybe_save_state(step, state),
            periodic_actions.PeriodicCallback(
                every_steps=self.write_metrics_interval_steps,
                callback_fn=_write_metrics,
            ),
            report_progress,
        ]

        train_step = self.train_step
        with (
            contextlib.ExitStack() as stack,
            clu_metric_writers.ensure_flushes(self.metric_writer),
        ):
            new_episode = True
            num_episode = 0
            for step in tqdm.trange(
                state.step * self.decision_frequency,
                num_env_steps,
                initial=state.step * self.decision_frequency,
                total=num_env_steps,
            ):
                if new_episode:
                    o_tm1, _ = train_env.reset()
                    num_episode += 1

                if step < self.warmup_steps:
                    a_tm1 = train_env.action_space.sample()
                else:
                    rng, sub = jax.random.split(rng)
                    a_tm1 = np.asarray(self.action(sub, state, o_tm1, train=True))

                o_t, r_tm1, d_t, t_t, _ = train_env.step(a_tm1)
                m_t = (not d_t) or t_t

                new_episode = d_t or t_t

                keep_sample = np.random.uniform() > max(
                    0, 1 - 1 / self.decision_frequency
                )

                if keep_sample or d_t:
                    self.buffer.insert(o_tm1, a_tm1, r_tm1, m_t, o_t)
                o_tm1 = o_t

                if (step >= self.warmup_steps) and (
                    (step + 1) % self.decision_frequency == 0
                ):
                    if self.buffer.size > 0:
                        rng, sub = jax.random.split(rng)
                        batch = self.buffer.sample(self.batch_size)
                        state = train_step(sub, state, batch)

                for callback in callbacks:
                    callback(step)

                if (state.step + 1) % self.eval_interval_steps == 0:
                    rng, sub = jax.random.split(rng)
                    metrics = self.eval(sub, state, _NUM_EVAL_EPISODES.value)
                    scalar_metrics = {
                        key: value.item()
                        for key, value in metrics.items()
                        if key in EVAL_SCALAR_TAGS
                    }
                    array_metrics = {
                        key: value
                        for key, value in metrics.items()
                        if key in EVAL_ARRAY_TAGS
                    }
                    text_metrics = {
                        key: str(value)
                        for key, value in metrics.items()
                        if key not in EVAL_SCALAR_TAGS
                    }
                    self.metric_writer.write_scalars(int(state.step), scalar_metrics)
                    self.metric_writer.write_scalars(
                        self.decision_frequency,
                        {
                            make_final_key(key): value
                            for (key, value) in scalar_metrics.items()
                        },
                    )
                    self.metric_writer.write_summaries(
                        self.decision_frequency, array_metrics
                    )
                    self.metric_writer.write_texts(
                        self.decision_frequency, text_metrics
                    )

        rng, sub = jax.random.split(rng)
        metrics = self.eval(sub, state, _FINAL_NUM_EVAL_EPISODES.value)

        scalar_metrics = {
            make_final_key(key): value.item()
            for key, value in metrics.items()
            if key in EVAL_SCALAR_TAGS
        }
        self.metric_writer.write_scalars(self.decision_frequency, scalar_metrics)
        o_tm1, _ = train_env.reset()
        if hasattr(self, "decision_dists"):
            import matplotlib.pyplot as plt

            from dsup.utils.plotting.plot_return_dists import plot_decision_dists

            action_conditioned_returns = self.decision_dists(state, o_tm1)
            fig = plot_decision_dists(action_conditioned_returns)
            self.metric_writer.write_images(0, {"decision_dists": fig})
            self.metric_writer.write_images(
                0, {"decision_dists_raw": np.array(action_conditioned_returns)}
            )
            plt.close(fig)
        self.maybe_save_state(self.num_steps, state, force=True)
