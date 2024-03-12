import logging
from typing import Any, Dict, Optional

from axolotl.prompt_strategies.chat_template import (
    ChatTemplatePrompter,
    ChatTemplateStrategy,
)
from axolotl.prompt_strategies.sharegpt import (
    ShareGPTPrompterV2,
    SimpleShareGPTPromptTokenizingStrategy,
)
from axolotl.utils.chat_templates import chat_templates
from transformers import PreTrainedTokenizer

logger = logging.getLogger("axolotl")


class OpenAIShareGPTPromptTokenizingStrategy(SimpleShareGPTPromptTokenizingStrategy):
    """
    Sharegpt strategy that remaps openai chat data to sharegpt format
    """

    def get_conversation_thread(self, prompt):
        conversations = prompt["messages"]
        role_key = "role"
        value_key = "content"
        role_map = {
            "user": "human",
            "human": "human",
            "assistant": "gpt",
            "gpt": "gpt",
            "system": "system",
        }
        turns = [{"from": role_map[t[role_key]], "value": t[value_key]} for t in conversations]
        return turns


def load_openai_sharegpt(tokenizer, cfg, ds_cfg: Optional[Dict[str, Any]] = None):
    conversation = ds_cfg["conversation"] if ds_cfg and "conversation" in ds_cfg else None
    strategy = OpenAIShareGPTPromptTokenizingStrategy(
        ShareGPTPrompterV2(
            conversation=conversation,
        ),
        tokenizer,
        cfg.train_on_inputs,
        cfg.sequence_len,
    )
    return strategy


# TODO (chiragjn): Add proper support for HF Tokenizers based chat templates
# Axolotl has provided an implementation but
# it requires the key to be "conversations" instead of "messages"
# secondly, it does not correctly mask the prompt tokens accounting for system prompt

# Goal is to have something like follows:
# def load_hf_chat_template(tokenizer: PreTrainedTokenizer, cfg, ds_cfg: Optional[Dict[str, Any]] = None):
#     chat_template = (
#         ds_cfg["chat_template"] if ds_cfg and "chat_template" in ds_cfg else None
#     )
#     if chat_template is None:
#         if not tokenizer.chat_template:
#             logger.warning("No chat template provided and tokenizer also does not have one set. Using default 'chatml'.")
#             chat_template = "chatml"
#     else:
#        chat_template = chat_templates(chat_template)
#     strategy = ChatTemplateStrategy(
#         ChatTemplatePrompter(tokenizer, chat_template),
#         tokenizer,
#         cfg.train_on_inputs,
#         cfg.sequence_len,
#     )
#     return strategy
