import pprint

import fiddle as fdl
import jax
from absl import app
from fiddle import absl_flags as fdl_flags
from fiddle import printing

from dsup import trainer  # noqa: F401
from dsup.utils import printing as custom_printing

AGENT = fdl_flags.DEFINE_fiddle_config(
    "agent", required=True, help_string="Fiddle config for agent construction"
)


def main(argv):
    agent_buildable = AGENT.value
    hparams = custom_printing.as_dict(
        agent_buildable, flatten_tree=True, buildable_fn_or_cls_key="name"
    )
    agent_trainer = fdl.build(agent_buildable)
    hparams["identifier"] = agent_trainer.identifier
    pprint.pp(hparams)
    agent_trainer.metric_writer.write_hparams(hparams)
    agent_trainer.train()


if __name__ == "__main__":
    jax.config.update("jax_numpy_rank_promotion", "raise")
    app.run(main)
