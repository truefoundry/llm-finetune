import json
import logging
import os
import shutil
import tempfile
from typing import Optional, Union

from transformers.trainer_utils import get_last_checkpoint

from mlfoundry_utils import (
    download_mlfoundry_artifact,
    get_checkpoint_artifact_version_with_step_or_none,
    get_latest_checkpoint_artifact_version_or_none,
)

logger = logging.getLogger("truefoundry-finetune")


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
            artifact_version_fqn=latest_checkpoint_artifact.fqn,
            download_dir=temp_dir,
            move_to=checkpoint_dir,
        )
    return checkpoint_dir


def download_checkpoint_of_step_if_present(
    ml_repo: str, checkpoint_artifact_name: str, step: int, local_dir: str
) -> Optional[str]:
    best_checkpoint_artifact = get_checkpoint_artifact_version_with_step_or_none(
        ml_repo=ml_repo, checkpoint_artifact_name=checkpoint_artifact_name, step=step
    )
    if not best_checkpoint_artifact:
        return
    logger.info(
        "Downloading best checkpoint from artifact version=%r step=%r because we might need at end",
        best_checkpoint_artifact.fqn,
        best_checkpoint_artifact.step,
    )
    os.makedirs(local_dir, exist_ok=True)
    checkpoint_dir = os.path.join(local_dir, f"checkpoint-{best_checkpoint_artifact.step}")
    with tempfile.TemporaryDirectory() as temp_dir:
        download_mlfoundry_artifact(
            artifact_version_fqn=best_checkpoint_artifact.fqn,
            download_dir=temp_dir,
            move_to=checkpoint_dir,
        )
    return checkpoint_dir


def get_best_checkpoint_for_resume_if_any(
    output_dir,
    last_checkpoint_dir: str,
    mlfoundry_enable_reporting: bool,
    mlfoundry_ml_repo: Optional[str],
    mlfoundry_checkpoint_artifact_name: Optional[str],
):
    trainer_state_file = os.path.join(last_checkpoint_dir, "trainer_state.json")
    if not os.path.exists(trainer_state_file):
        return None
    try:
        with open(trainer_state_file) as trainer_state_f:
            trainer_state = json.load(trainer_state_f)
        if "best_model_checkpoint" not in trainer_state:
            return None

        best_checkpoint_name = os.path.basename(trainer_state["best_model_checkpoint"])
        last_checkpoint_name = os.path.basename(last_checkpoint_dir)

        if best_checkpoint_name != last_checkpoint_name:
            best_checkpoint_dir = os.path.join(output_dir, best_checkpoint_name)
            if os.path.exists(best_checkpoint_dir):
                return best_checkpoint_dir

            if mlfoundry_enable_reporting and mlfoundry_checkpoint_artifact_name:
                best_step = int(best_checkpoint_name.split("checkpoint-", 1)[1])
                logger.info(f"Checking for checkpoint with step {best_step} from same job run...")
                best_checkpoint_dir = download_checkpoint_of_step_if_present(
                    ml_repo=mlfoundry_ml_repo,
                    checkpoint_artifact_name=mlfoundry_checkpoint_artifact_name,
                    step=best_step,
                    local_dir=output_dir,
                )
                return best_checkpoint_dir
    except Exception as e:
        logger.warning(f"Unable to get the best checkpoint, error: {e}")

    return None


def get_last_checkpoint_for_resume_if_any(
    output_dir,
    resume_from_checkpoint: Optional[Union[bool, str]],
    mlfoundry_enable_reporting: bool,
    mlfoundry_ml_repo: Optional[str],
    mlfoundry_checkpoint_artifact_name: Optional[str],
) -> Optional[str]:
    last_checkpoint_dir = None
    check_mlfoundry = False
    # resume_from_checkpoint can be None/true/false/string, None is default
    if resume_from_checkpoint is None:
        # If no explicit choice has been made we will try and check with mlfoundry we are allowed to
        check_mlfoundry = True
    elif isinstance(resume_from_checkpoint, str):
        # If an explicit choice has been made we will check if the checkpoint exists on disk
        if os.path.exists(resume_from_checkpoint):
            last_checkpoint_dir = resume_from_checkpoint
        else:
            raise ValueError(f"Provided path for --resume_from_checkpoint `{resume_from_checkpoint}` does not exist!")
        # TODO (chiragjn): Add support for resuming from an already saved checkpoint outside of the job run
        #   Although this is risky, because all other args (model, data, state) should remain same for a "correct" resume
        #   Note: Instead if we just want to resume from last checkpoint of the same job run then just use --mlfoundry_enable_reporting true --mlfoundry_checkpoint_artifact_name <name>
        # elif _is_mlfoundry_artifact(training_arguments.resume_from_checkpoint):
        #     _download_mlfoundry_artifact(...)
    elif resume_from_checkpoint is True:
        # If set to true, we will automatically locate the latest checkpoint, first checking output dir, next mlfoundry if we are allowed to
        if os.path.exists(output_dir):
            possible_last_checkpoint_dir = get_last_checkpoint(output_dir)
            if possible_last_checkpoint_dir:
                last_checkpoint_dir = possible_last_checkpoint_dir

        if not last_checkpoint_dir:
            check_mlfoundry = True

    if check_mlfoundry and mlfoundry_enable_reporting and mlfoundry_checkpoint_artifact_name:
        logger.info("Checking for any past checkpoints from same job run...")
        last_checkpoint_dir = download_last_checkpoint_if_present(
            ml_repo=mlfoundry_ml_repo,
            checkpoint_artifact_name=mlfoundry_checkpoint_artifact_name,
            local_dir=output_dir,
        )

    if last_checkpoint_dir:
        _ = get_best_checkpoint_for_resume_if_any(
            output_dir=output_dir,
            last_checkpoint_dir=last_checkpoint_dir,
            mlfoundry_enable_reporting=mlfoundry_enable_reporting,
            mlfoundry_ml_repo=mlfoundry_ml_repo,
            mlfoundry_checkpoint_artifact_name=mlfoundry_checkpoint_artifact_name,
        )

    return last_checkpoint_dir


def cleanup_checkpoints(
    output_dir: str,
):
    logger.info("Cleaning up older checkpoints...")
    for f in os.listdir(output_dir):
        f_path = os.path.join(output_dir, f)
        if os.path.isdir(f_path) and f.startswith("checkpoint-"):
            shutil.rmtree(f_path)
