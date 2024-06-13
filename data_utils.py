import enum
import logging
import os

from mlfoundry_utils import (
    download_mlfoundry_artifact,
    is_mlfoundry_artifact,
    sanitize_name,
)

logger = logging.getLogger("axolotl")


def find_all_jsonl_files(path):
    for root, dirs, files in os.walk(path):
        for file in files:
            filepath = os.path.join(root, file)
            filename = os.path.basename(filepath)
            if filename.endswith(".jsonl") and not filename.startswith("."):
                yield filepath


class DatasetType(str, enum.Enum):
    completion = "completion"
    chat = "chat"


def _make_dataset_file_source(
    path,
    split="train",
    dataset_type: DatasetType = DatasetType.completion,
):
    """
    Axolotl dynamically loads prompt strategies based on the `type` key
    The modules are present at axolotl.prompt_strategies.*
    The `load` function in the module is called with the tokenizer, cfg and ds_cfg

    Ideally we want to use the HF tokenizers library to apply the base model's chat template
    But axolotl's chat template strategy forces to select one of the built-in template.
    """
    if dataset_type == DatasetType.completion:
        return {
            "path": path,
            "ds_type": "json",
            "type": {
                "system_prompt": "",
                "field_system": "system",
                "field_instruction": "prompt",
                "field_output": "completion",
                "format": "{instruction}\n{input}\n",
                "no_input_format": "{instruction}\n",
                "system_format": "{system}\n",
            },
            "split": split,
        }
    elif dataset_type == DatasetType.chat:
        return {
            "path": path,
            "ds_type": "json",
            "type": "chat_template",
            "field_messages": "messages",
            "message_field_role": "role",
            "message_field_content": "content",
            "roles": {"system": ["system"], "user": ["user", "human"], "assistant": ["assistant"], "tool": ["tool"]},
            "split": split,
        }
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")


def dataset_uri_to_axolotl_datasources(
    uri,
    download_dir,
    dataset_type: DatasetType = DatasetType.completion,
):
    # TODO: Add support for HF datasets
    if uri.startswith("https://"):
        return [_make_dataset_file_source(path=uri, dataset_type=dataset_type)]
    elif is_mlfoundry_artifact(uri):
        datasources = []
        logger.info("Downloading artifact from mlfoundry")
        artifact_download_dir = os.path.join(download_dir, sanitize_name(uri))
        download_path = download_mlfoundry_artifact(
            artifact_version_fqn=uri, download_dir=artifact_download_dir, overwrite=True
        )
        for filepath in find_all_jsonl_files(download_path):
            logger.info("Adding jsonl file {filepath}")
            datasources.append(_make_dataset_file_source(path=filepath, dataset_type=dataset_type))
        return datasources
    elif os.path.exists(uri):
        datasources = []
        if os.path.isdir(uri):
            for filepath in find_all_jsonl_files(uri):
                datasources.append(_make_dataset_file_source(path=filepath, dataset_type=dataset_type))
        else:
            datasources = [_make_dataset_file_source(path=uri, dataset_type=dataset_type)]
        return datasources
    else:
        raise ValueError("Unsupported data uri or path does not exist: {uri}")


# --- Reference Notes ---


# Other axolotl strategies:
#     # Llama 2 chat template with data that looks like
#     {"conversations": [{"role": "system", "content": "You are a helpful assistant"}, {"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]}

#     Config:
#     {
#         "type": "sharegpt",
#         "conversation": "llama-2",
#         "field_human": "user",
#         "field_model": "assistant",
#     }

#     # HF Tokenizers Chat Template
#     {"conversations": [{"role": "system", "content": "You are a helpful assistant"}, {"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]}

#     {
#         "type": "chat_template",
#         "chat_template": "chatml",
#     }
