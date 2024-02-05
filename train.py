import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import fire
from axolotl.cli import load_cfg
from axolotl.cli.train import do_train
from axolotl.common.cli import TrainerCliArgs
from transformers.hf_argparser import HfArgumentParser

LOG = logging.getLogger("axolotl.cli.train")


@dataclass
class TrueFoundryArguments:
    train_data_uri: str = field(metadata={"help": "URL to the jsonl training dataset"})
    eval_data_uri: Optional[str] = field(
        default="NA",
        metadata={"help": "URL to the jsonl evaluation dataset. Overrides eval_size. Leave as NA if not available"},
    )
    mlfoundry_enable_reporting: bool = field(
        default=False,
        metadata={"help": "Use mlfoundry to log metrics, checkpoints and model"},
    )
    mlfoundry_ml_repo: Optional[str] = field(default=None, metadata={"help": "ML Repo to put the model to"})
    mlfoundry_run_name: Optional[str] = field(
        default=None,
        metadata={"help": "Run name for mlfoundry run"},
    )
    mlfoundry_log_checkpoints: bool = field(
        default=True,
        metadata={"help": "If to log intermediate checkpoints to mlfoundry"},
    )
    mlfoundry_checkpoint_artifact_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "ML Repo artifact name to save checkpoints. \n"
            "The artifact will be created if it does not exist under the give ML Repo"
        },
    )
    cleanup_output_dir_on_start: bool = field(
        default=False,
        metadata={"help": "Cleanup output dir at the start of training run"},
    )


def train_with_truefoundry(config: Path = Path("examples/"), **kwargs):
    parsed_cfg = load_cfg(config, **kwargs)
    parser = HfArgumentParser((TrueFoundryArguments, TrainerCliArgs))
    tfy_args, parsed_cli_args, _ = parser.parse_args_into_dataclasses(return_remaining_strings=True)
    return do_train(parsed_cfg, parsed_cli_args)


# import axolotl.train
# def patched_pretrain_hooks(cfg, trainer):
#     pass
# if hasattr(axolotl.train, "pretrain_hooks"):
#     axolotl.train.pretrain_hooks = patched_pretrain_hooks
# else:
#     raise ValueError(
#         "Did not find `pretrain_hooks` on `axolotl.train`. "
#         "This is required to patch and add callbacks"
#     )

if __name__ == "__main__":
    fire.Fire(train_with_truefoundry)
