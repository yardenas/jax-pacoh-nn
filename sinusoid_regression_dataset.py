from typing import Iterator, Tuple

import numpy as np


class SinusoidRegression:
    def __init__(
        self,
        meta_batch_size: int,
        num_train_shots: int,
        num_test_shots: int,
        seed: int = 666,
    ):
        self.meta_batch_size = meta_batch_size
        self.num_train_shots = num_train_shots
        self.num_test_shots = num_test_shots
        self.rs = np.random.RandomState(seed)

    @property
    def train_set(
        self,
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        while True:
            yield self._make_batch(self.num_train_shots)[0]

    @property
    def test_set(
        self,
    ) -> Iterator[Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]]:
        while True:
            yield self._make_batch(self.num_test_shots)

    def _make_batch(self, num_shots: int) -> Tuple[np.ndarray, np.ndarray]:
        # Select amplitude and phase for the task
        amplitudes = []
        phases = []
        for _ in range(self.meta_batch_size):
            amplitudes.append(self.rs.uniform(low=0.1, high=0.5))
            phases.append(self.rs.uniform(low=0.0, high=np.pi))

        def get_batch(num_shots: int) -> Tuple[np.ndarray, np.ndarray]:
            xs, ys = [], []
            for amplitude, phase in zip(amplitudes, phases):
                if num_shots > 0:
                    x = self.rs.uniform(low=-5.0, high=5.0, size=(num_shots, 1))
                else:
                    x = np.linspace(-5.0, 5.0, 1000)[:, None]
                y = amplitude * np.sin(x + phase)
                xs.append(x)
                ys.append(y)
            return np.stack(xs), np.stack(ys)

        (x1, y1), (x2, y2) = get_batch(num_shots), get_batch(-1)
        return (x1, y1), (x2, y2)
