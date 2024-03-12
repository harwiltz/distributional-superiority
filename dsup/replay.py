from typing import Union

import gymnasium as gym
import numpy as np

from dsup.types import TransitionBatch

State = np.ndarray
Action = Union[np.ndarray[float], int]


class CircularReplayBuffer:
    def __init__(
        self,
        capacity: int,
    ):
        self.capacity = capacity
        self.o_tm1 = None
        self.a_tm1 = None
        self.r_tm1 = None
        self.m_t = None
        self.o_t = None

        self._initialized = False

        self._size = 0
        self._ptr = 0

    def _allocate_buffers(
        self, o_tm1: State, a_tm1: Action, r_tm1: float, m_t: float, o_t: State
    ) -> None:
        self.o_tm1 = np.empty((self.capacity, *o_tm1.shape), dtype=o_tm1.dtype)
        self.a_tm1 = np.empty((self.capacity, *a_tm1.shape), dtype=a_tm1.dtype)
        self.r_tm1 = np.empty((self.capacity,), dtype=np.float32)
        self.m_t = np.empty((self.capacity,), dtype=np.float32)
        self.o_t = np.empty((self.capacity, *o_t.shape), dtype=o_t.dtype)
        self._initialized = True

    @property
    def size(self):
        return self._size

    def insert(self, o_tm1: State, a_tm1: Action, r_tm1: float, m_t: float, o_t: State):
        if not self._initialized:
            self._allocate_buffers(o_tm1, a_tm1, r_tm1, m_t, o_t)
        self.o_tm1[self._ptr] = o_tm1
        self.a_tm1[self._ptr] = a_tm1
        self.r_tm1[self._ptr] = r_tm1
        self.m_t[self._ptr] = m_t
        self.o_t[self._ptr] = o_t
        self._ptr = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> TransitionBatch:
        idx = np.random.randint(self.size, size=batch_size)
        return TransitionBatch(
            o_tm1=self.o_tm1[idx],
            a_tm1=self.a_tm1[idx],
            r_tm1=self.r_tm1[idx],
            m_t=self.m_t[idx],
            o_t=self.o_t[idx],
        )
