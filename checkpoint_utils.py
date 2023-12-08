import json
import logging
import os
import shutil
import tempfile
from typing import Optional, Union

from accelerate.state import AcceleratorState
from transformers.trainer_utils import get_last_checkpoint

from mlfoundry_utils import (
    download_mlfoundry_artifact,
    get_latest_checkpoint_artifact_version_or_none,
)

logger = logging.getLogger("truefoundry-finetune")


# --- Model checkpointing, saving and logging utils ---
def download_last_checkpoint_if_present(ml_repo: str, checkpoint_artifact_name: str, local_dir: str) -> Optional[str]:
    latest_checkpoint_artifact = get_latest_checkpoint_artifact_version_or_none(
        ml_repo=ml_repo, checkpoint_artifact_name=checkpoint_artifact_name
    )
    if not latest_checkpoint_artifact:
        return
    logger.info(
        "Downloading last checkpoint from artifact version=%r step=%r to resume training",
        latest_checkpoint_artifact.fqn,
        latest_checkpoint_artifact.step,
    )
    os.makedirs(local_dir, exist_ok=True)
    checkpoint_dir = os.path.join(local_dir, f"checkpoint-{latest_checkpoint_artifact.step}")
    with tempfile.TemporaryDirectory() as temp_dir:
        download_mlfoundry_artifact(
            artifact_version_fqn=latest_checkpoint_artifact.fqn, download_dir=temp_dir, move_to=checkpoint_dir
        )
    return checkpoint_dir


def get_checkpoint_for_resume_if_any(
    cache_dir,
    output_dir,
    resume_from_checkpoint: Optional[Union[bool, str]],
    mlfoundry_enable_reporting: bool,
    mlfoundry_ml_repo: Optional[str],
    mlfoundry_checkpoint_artifact_name: Optional[str],
) -> Optional[str]:
    accelerator_s = AcceleratorState()
    last_checkpoint_info_path = os.path.join(cache_dir, "last_checkpoint_info.json")
    last_checkpoint_dir = None
    if accelerator_s.is_main_process:
        check_mlfoundry = False
        # resume_from_checkpoint can be None/true/false/string, None is default
        if resume_from_checkpoint is None:
            check_mlfoundry = True
        elif isinstance(resume_from_checkpoint, str):
            if os.path.exists(resume_from_checkpoint):
                last_checkpoint_dir = resume_from_checkpoint

            # TODO (chiragjn): Add support for resuming from an already saved checkpoint outside of the job run
            #   Although this is risky, because all other args (model, data, state) should remain same for a "correct" resume
            #   Note: Instead if we just want to resume from last checkpoint of the same job run then just use --mlfoundry_enable_reporting true --mlfoundry_checkpoint_artifact_name <name>
            # elif _is_mlfoundry_artifact(training_arguments.resume_from_checkpoint):
            #     _download_mlfoundry_artifact(...)

        elif resume_from_checkpoint is True:
            # Try locating latest checkpoint from output dir first
            if os.path.exists(output_dir):
                possible_last_checkpoint_dir = get_last_checkpoint(output_dir)
                if possible_last_checkpoint_dir:
                    last_checkpoint_dir = possible_last_checkpoint_dir

            if not last_checkpoint_dir:
                check_mlfoundry = True

        if check_mlfoundry and mlfoundry_enable_reporting:
            logger.info("Checking for any past checkpoints from same job run...")
            if mlfoundry_checkpoint_artifact_name:
                last_checkpoint_dir = download_last_checkpoint_if_present(
                    ml_repo=mlfoundry_ml_repo,
                    checkpoint_artifact_name=mlfoundry_checkpoint_artifact_name,
                    local_dir=output_dir,
                )

        with open(last_checkpoint_info_path, "w") as f:
            last_checkpoint_info = {"last_checkpoint_dir": last_checkpoint_dir}
            json.dump(last_checkpoint_info, f)

        # if last_checkpoint_dir:
        #     # if we have the last checkpoint
        #     trainer_state_file = os.path.join(last_checkpoint_dir, "trainer_state.json")
        #     if os.path.exists(trainer_state_file):
        #         try:
        #             with open(trainer_state_file) as trainer_state_f:
        #                 trainer_state = json.load(trainer_state_f)
        #             if "best_model_checkpoint" in trainer_state:
        #                 best_model_checkpoint_name = os.path.basename(trainer_state["best_model_checkpoint"])
        #                 best_model_checkpoint_name.split("checkpoint-", 1)
        #         except Exception:
        #             raise NotImplementedError
    else:
        with open(last_checkpoint_info_path, "r") as f:
            last_checkpoint_info = json.load(f)
        last_checkpoint_dir = last_checkpoint_info["last_checkpoint_dir"]
    return last_checkpoint_dir


def cleanup_checkpoints(
    output_dir: str,
):
    logger.info("Cleaning up older checkpoints...")
    for f in os.listdir(output_dir):
        f_path = os.path.join(output_dir, f)
        if os.path.isdir(f_path) and f.startswith("checkpoint-"):
            shutil.rmtree(f_path)
