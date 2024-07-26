import copy
import logging
import math
import os
import random
import re
import shutil
import string
from typing import Any, Dict, Optional

import numpy as np
from huggingface_hub import scan_cache_dir
from transformers import TrainerCallback
from transformers.integrations import rewrite_logs
from truefoundry import ml as mlfoundry

logger = logging.getLogger("axolotl")

MLFOUNDRY_ARTIFACT_PREFIX = "artifact:"
TFY_INTERNAL_JOB_NAME = os.getenv("TFY_INTERNAL_COMPONENT_NAME")
TFY_INTERNAL_JOB_RUN_NAME = os.getenv("TFY_INTERNAL_JOB_RUN_NAME")


def _drop_non_finite_values(dct: Dict[str, Any]) -> Dict[str, Any]:
    sanitized = {}
    for k, v in dct.items():
        if isinstance(v, (int, float, np.integer, np.floating)):
            if not math.isfinite(v):
                logger.warning(f"Dropping non-finite value for key={k} value={v!r}")
                continue
        sanitized[k] = v
    return sanitized


def is_mlfoundry_artifact(value: str):
    # TODO (chiragjn): This should be made more strict
    if value.startswith(MLFOUNDRY_ARTIFACT_PREFIX):
        return True


def download_mlfoundry_artifact(
    artifact_version_fqn: str, download_dir: str, overwrite: bool = False, move_to: Optional[str] = None
):
    client = mlfoundry.get_client()
    artifact_version = client.get_artifact_version_by_fqn(artifact_version_fqn)
    os.makedirs(download_dir, exist_ok=True)
    files_dir = artifact_version.download(download_dir, overwrite=overwrite)
    if move_to:
        files_dir = shutil.move(files_dir, move_to)
    return files_dir


def log_model_to_mlfoundry(
    run: mlfoundry.MlFoundryRun,
    model_name: str,
    model_dir: str,
    hf_hub_model_id: str,
    metadata: Optional[Dict[str, Any]] = None,
    step: Optional[int] = None,
):
    metadata = metadata or {}
    logger.info("Uploading Model...")
    hf_cache_info = scan_cache_dir()
    files_to_save = []
    for repo in hf_cache_info.repos:
        if repo.repo_id == hf_hub_model_id:
            for revision in repo.revisions:
                for file in revision.files:
                    if file.file_path.name.endswith(".py"):
                        files_to_save.append(file.file_path)
                break

    # copy the files to output_dir of pipeline
    for file_path in files_to_save:
        match = re.match(r".*snapshots\/[^\/]+\/(.*)", str(file_path))
        if match:
            relative_path = match.group(1)
            destination_path = os.path.join(model_dir, relative_path)
            os.makedirs(os.path.dirname(destination_path), exist_ok=True)
            shutil.copy(str(file_path), destination_path)
        else:
            logger.warning("Python file in hf model cache in unknown path:", file_path)

    metadata.update(
        {
            "pipeline_tag": "text-generation",
            "library_name": "transformers",
            "base_model": hf_hub_model_id,
            "huggingface_model_url": f"https://huggingface.co/{hf_hub_model_id}",
        }
    )
    metadata = _drop_non_finite_values(metadata)
    run.log_model(
        name=model_name,
        model_file_or_folder=model_dir,
        framework=mlfoundry.ModelFramework.TRANSFORMERS,
        metadata=metadata,
        step=step or 0,
    )


def get_latest_checkpoint_artifact_version_or_none(
    ml_repo: str,
    checkpoint_artifact_name: str,
) -> Optional[mlfoundry.ArtifactVersion]:
    # TODO (chiragjn):  Reduce coupling with checkpointing, log lines are still related
    latest_checkpoint_artifact = None
    try:
        client = mlfoundry.get_client()
        artifact_versions = client.list_artifact_versions(ml_repo=ml_repo, name=checkpoint_artifact_name)
        latest_checkpoint_artifact = next(artifact_versions)
    except StopIteration:
        logger.info(
            f"No previous checkpoints found at artifact={checkpoint_artifact_name!r} in ml_repo={ml_repo!r}",
        )
    # TODO: We should have specific exception to identify if the artifact does not exist
    except Exception as e:
        logger.info("No previous checkpoints found. Message=%s", e)

    return latest_checkpoint_artifact


def get_checkpoint_artifact_version_with_step_or_none(
    ml_repo: str, checkpoint_artifact_name: str, step: int
) -> Optional[mlfoundry.ArtifactVersion]:
    checkpoint_artifact_version_with_step = None
    try:
        client = mlfoundry.get_client()
        artifact_versions = client.list_artifact_versions(ml_repo=ml_repo, name=checkpoint_artifact_name)
        for artifact_version in artifact_versions:
            if artifact_version.step == step:
                checkpoint_artifact_version_with_step = artifact_version
                break
    except Exception as e:
        logger.warning(f"No checkpoint found for step {step}. Message=%s", e)

    return checkpoint_artifact_version_with_step


class MLFoundryCallback(TrainerCallback):
    def __init__(
        self,
        run: mlfoundry.MlFoundryRun,
        log_checkpoints: bool = True,
        checkpoint_artifact_name: Optional[str] = None,
    ):
        self._run = run
        self._checkpoint_artifact_name = checkpoint_artifact_name
        self._log_checkpoints = log_checkpoints

        if not self._checkpoint_artifact_name:
            logger.warning("checkpoint_artifact_name not passed. Checkpoints will not be logged to MLFoundry")

    # noinspection PyMethodOverriding
    def on_log(self, args, state, control, logs, model=None, **kwargs):
        if not state.is_world_process_zero:
            return

        metrics = {}
        for k, v in logs.items():
            if k.startswith("system/gpu"):
                continue
            if isinstance(v, (int, float, np.integer, np.floating)) and math.isfinite(v):
                metrics[k] = v
            else:
                logger.warning(
                    f'Trainer is attempting to log a value of "{v}" of'
                    f' type {type(v)} for key "{k}" as a metric.'
                    " Mlfoundry's log_metric() only accepts finite float and"
                    " int types so we dropped this attribute."
                )
        self._run.log_metrics(rewrite_logs(metrics), step=state.global_step)

    def on_save(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return

        if not self._log_checkpoints:
            return

        if not self._checkpoint_artifact_name:
            return

        ckpt_dir = f"checkpoint-{state.global_step}"
        artifact_path = os.path.join(args.output_dir, ckpt_dir)
        description = None
        if TFY_INTERNAL_JOB_NAME:
            description = f"Checkpoint from finetuning job={TFY_INTERNAL_JOB_NAME} run={TFY_INTERNAL_JOB_RUN_NAME}"
        logger.info(f"Uploading checkpoint {ckpt_dir} ...")
        metadata = {}
        for log in state.log_history:
            if isinstance(log, dict) and log.get("step") == state.global_step:
                metadata = log.copy()
        metadata = _drop_non_finite_values(metadata)
        self._run.log_artifact(
            name=self._checkpoint_artifact_name,
            artifact_paths=[(artifact_path,)],
            metadata=metadata,
            step=state.global_step,
            description=description,
        )


def sanitize_name(value):
    return re.sub(rf"[{re.escape(string.punctuation)}]+", "-", value.encode("ascii", "ignore").decode("utf-8"))


def generate_run_name(model_id, seed: Optional[int] = None):
    *_, model_name = model_id.split("/", 1)
    sanitized_model_name = sanitize_name(model_name)
    alphabet = string.ascii_lowercase + string.digits
    rng = random.Random(seed) if seed is not None else random
    random_id = "".join(rng.choices(alphabet, k=6))
    run_name = f"ft-{sanitized_model_name}-{random_id}"
    return run_name


def get_or_create_run(ml_repo: str, run_name: str, auto_end: bool = False, create_ml_repo: bool = False):
    client = mlfoundry.get_client()
    if create_ml_repo:
        client.create_ml_repo(ml_repo=ml_repo)
    try:
        run = client.get_run_by_name(ml_repo=ml_repo, run_name=run_name)
    except Exception as e:
        if "RESOURCE_DOES_NOT_EXIST" not in str(e):
            raise
        run = client.create_run(ml_repo=ml_repo, run_name=run_name, auto_end=auto_end)
    return run


def maybe_log_params_to_mlfoundry(run: mlfoundry.MlFoundryRun, params: Dict[str, Any]):
    if not params:
        return
    if run.get_params():
        logger.warning("Skipping logging params because they already exist")
    else:
        params = copy.deepcopy(params)
        batch_size = 50
        items = list(params.items())
        for idx in range(0, len(items), batch_size):
            mini_batch = dict(items[idx : idx + batch_size])
            run.log_params(mini_batch, flatten_params=False)
