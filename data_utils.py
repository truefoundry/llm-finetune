import copy
import json
import logging
import os
import tempfile
from typing import Optional
from urllib.parse import parse_qsl, urlparse

import torch
from accelerate.state import AcceleratorState
from cloudfiles import CloudFile
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split

from mlfoundry_utils import download_mlfoundry_artifact, is_mlfoundry_artifact

logger = logging.getLogger("truefoundry-finetune")

PROMPT_KEY = "prompt"
COMPLETION_KEY = "completion"
IGNORE_INDEX = -100  # -100 is the default ignore index in CrossEntropyLoss


class DataValidationException(Exception):
    pass


def get_data_from_snowflake_table(
    uri: str,
    max_num_samples: int = 0,
    batch_size=500,
):
    """
    URI format:
    snowflake://{user}:{password}@{account}/{database}/{schema}/{table}?warehouse={warehouse}&role={role}
    """
    import snowflake.connector

    parsed_uri = urlparse(uri)
    database, schema, table, *_ = parsed_uri.path.strip("/").split("/")
    query_params = dict(parse_qsl(parsed_uri.query))
    kwargs = dict(
        user=parsed_uri.username,
        password="*******",
        account=parsed_uri.hostname,
        database=database,
        schema=schema,
        **query_params,
    )
    logger.info(f"Connecting to Snowflake table {table} with args: {kwargs}")
    kwargs["password"] = parsed_uri.password
    connection, cursor = None, None
    try:
        connection = snowflake.connector.connect(**kwargs)
        cursor = connection.cursor()
        if max_num_samples > 0:
            cursor.execute(f"SELECT * FROM {table} LIMIT {max_num_samples}")
        else:
            cursor.execute(f"SELECT * FROM {table}")
        column_names = [column_name.name for column_name in cursor.description]
        print(f"Got columns: {column_names}")
        while True:
            rows = cursor.fetchmany(batch_size)
            if not rows:
                break
            for row in rows:
                line = dict(zip(column_names, row))
                line = {
                    PROMPT_KEY: line.get(PROMPT_KEY) or line.get("PROMPT"),
                    COMPLETION_KEY: line.get(COMPLETION_KEY) or line.get("COMPLETION"),
                }
                yield json.dumps(line)
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()


def _read_lines_from_files(download_path):
    for root, dirs, files in os.walk(download_path):
        for file in files:
            filepath = os.path.join(root, file)
            filename = os.path.basename(filepath)
            if filename.endswith(".jsonl") and not filename.startswith("."):
                logger.info(f"Loading file {filename} ...")
                with open(filepath) as f:
                    for line in f.readlines():
                        yield line


def _read_lines_from_cloudfile(path):
    raw_data = CloudFile(path).get().decode("utf-8").split("\n")
    for line in raw_data:
        yield line


def load_data(path, max_num_samples: Optional[int] = None):
    data = []
    n = max_num_samples if max_num_samples else -1
    count = 0
    with tempfile.TemporaryDirectory() as download_dir:
        if is_mlfoundry_artifact(path):
            logger.info("Downloading artifact from mlfoundry")
            download_path = download_mlfoundry_artifact(artifact_version_fqn=path, download_dir=download_dir)
            lines = _read_lines_from_files(download_path)
        elif path.startswith("snowflake://"):
            logger.info(f"Loading data from snowflake db ...")
            lines = get_data_from_snowflake_table(uri=path, max_num_samples=max_num_samples)
        else:
            logger.info(f"Loading data from link: {path}")
            lines = _read_lines_from_cloudfile(path)
        for line_no, line in enumerate(lines, start=1):
            if n > 0 and count >= n:
                break
            if not line.strip():
                continue
            try:
                datapoint_dict = json.loads(line)
            except json.decoder.JSONDecodeError as je:
                raise DataValidationException(
                    f"Failed to parse json line on line number {line_no}. Line: {line[:150]}..."
                ) from je
            else:
                for key in (PROMPT_KEY, COMPLETION_KEY):
                    if key not in datapoint_dict:
                        raise DataValidationException(
                            f"Required key `{key}` is missing from json line on line number {line_no}. Line: {line[:150]}..."
                        )
                    if not isinstance(datapoint_dict[key], str):
                        raise DataValidationException(
                            f"Value for `{key}` is not string on line number {line_no}. Line: {line[:150]}..."
                        )

                datapoint_dict = {
                    PROMPT_KEY: datapoint_dict[PROMPT_KEY],
                    COMPLETION_KEY: datapoint_dict[COMPLETION_KEY],
                }
                data.append(datapoint_dict)
                count += 1
    return data


def get_data(
    train_data: str,
    eval_data: Optional[str],
    eval_size: float = 0.1,
    max_num_samples: int = 0,
    data_seed: int = 42,
):
    logger.info(f"Loading train dataset ...")
    train_data = load_data(train_data, max_num_samples=max_num_samples)
    if eval_data and eval_data != "NA":
        logger.info(f"Loading eval dataset {eval_data}...")
        eval_data = load_data(eval_data, max_num_samples=max_num_samples)
    elif eval_size:
        logger.info(f"No eval dataset given, splitting from training dataset...")
        train_data, eval_data = train_test_split(
            train_data,
            test_size=eval_size,
            random_state=data_seed,
        )
    return train_data, eval_data


class DatasetBuilder:
    """Dataset agnostic class to take in input_ids and labels and spit out tokens"""

    def __init__(self, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def batch_tokenize(self, texts):
        """Tokenizes text. Presently doesn't pad inputs, just returns input ids."""
        tokenized = [
            self.tokenizer(
                text,
                return_tensors="pt",
                padding="longest",
                max_length=self.max_length,
                truncation=True,
            ).input_ids
            for text in texts
        ]
        return tokenized

    def construct_dataset(self, input_batch):
        tokenized_input_ids = self.batch_tokenize(input_batch[PROMPT_KEY])
        tokenized_labels = self.batch_tokenize(input_batch[COMPLETION_KEY])
        return {"input_ids": tokenized_input_ids, "labels": tokenized_labels}


class CausalDatasetBuilder(DatasetBuilder):
    """Builds generative dataset for Causal LM."""

    def __init__(self, tokenizer, max_length, train_on_prompt=True):
        super().__init__(tokenizer, max_length)
        self.train_on_prompt = train_on_prompt

    def construct_dataset(self, input_batch):
        labels = []
        for prompt, completion in zip(input_batch[PROMPT_KEY], input_batch[COMPLETION_KEY]):
            labels.append(prompt + "\n" + completion + self.tokenizer.eos_token)
        input_ids = [val.squeeze() for val in self.batch_tokenize(labels)]
        labels = copy.deepcopy(input_ids)
        if not self.train_on_prompt:
            tokenized_prompts = self.batch_tokenize(input_batch[PROMPT_KEY])
            prompt_lens = [val.shape[1] for val in tokenized_prompts]
            for label, source_len in zip(labels, prompt_lens):
                label[:source_len] = IGNORE_INDEX
        return {"input_ids": input_ids, "labels": labels}


class SequenceDataCollator:
    """Collate examples for dynamic batch construction in supervised fine-tuning."""

    def __init__(self, tokenizer, multiple_of=None):
        self.tokenizer = tokenizer
        self.multiple_of = multiple_of

    def pad_to_multiple(self, tensor, value):
        multiple = self.multiple_of
        n = tensor.size(-1)
        target_length = (n + multiple - 1) // multiple * multiple
        return torch.nn.functional.pad(tensor, (0, target_length - n), value=value)

    def __call__(self, instances):
        input_ids = [instance["input_ids"] for instance in instances]
        labels = [instance["labels"] for instance in instances]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )  # -100 tells torch to ignore these tokens in loss computation.
        if self.multiple_of:
            input_ids = self.pad_to_multiple(input_ids, value=self.tokenizer.pad_token_id)
            labels = self.pad_to_multiple(labels, value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def build_dataset(
    train_data,
    eval_data,
    tokenizer,
    max_length: int,
    train_on_prompt: bool,
    cache_dir: str,
):
    accelerator_s = AcceleratorState()
    logger.info("Building dataset...")
    dataset_cache_path = os.path.join(cache_dir, "dataset")
    if accelerator_s.is_main_process:
        builder = CausalDatasetBuilder(tokenizer=tokenizer, max_length=max_length, train_on_prompt=train_on_prompt)
        dataset_dict = DatasetDict(train=Dataset.from_list(train_data), eval=Dataset.from_list(eval_data))
        # TODO (chiragjn): Read cpu limits from cgroup, cpu_count is not usable in containers environment
        num_proc = max(1, min(4, os.cpu_count()))
        num_proc = num_proc if num_proc > 1 else None
        dataset_dict = dataset_dict.map(
            builder.construct_dataset,
            remove_columns=[PROMPT_KEY, COMPLETION_KEY],
            batched=True,
            batch_size=32,
            num_proc=num_proc,
        )
        dataset_dict.save_to_disk(dataset_cache_path)
    else:
        logger.info("Loading datasets from cache ...")
        dataset_dict = DatasetDict.load_from_disk(dataset_cache_path)
    dataset_dict = dataset_dict.with_format("torch")
    train_dataset, eval_dataset = dataset_dict["train"], dataset_dict["eval"]
    logger.info(f"Train data size: {len(train_dataset)}")
    logger.info(f"Eval data size: {len(eval_dataset)}")
    return train_dataset, eval_dataset
