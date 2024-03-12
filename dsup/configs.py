import dataclasses
from typing import Iterator

import fiddle as fdl
import jax
from etils import epath


@dataclasses.dataclass
class Runner:
    x: str
    seed: int

    def run(self, dir: epath.Path):
        rng = jax.random.PRNGKey(self.seed)
        y = jax.random.randint(rng, minval=0, maxval=10, shape=(4,))
        print("Hello, my name is", self.x)
        print("Random samples are", y)
        (dir / f"test-{self.seed}.txt").write_text(self.x)


def base() -> fdl.Config[Runner]:
    return fdl.Config(Runner, x="x", seed=42)


def sweep_x(root: fdl.Config[Runner]) -> Iterator[fdl.Config[Runner]]:
    for x in ["x", "y"]:
        yield fdl.copy_with(root, x=x)


def sweep_seed(root: fdl.Config[Runner]) -> Iterator[fdl.Config[Runner]]:
    for seed in range(3):
        yield fdl.copy_with(root, seed=seed)
