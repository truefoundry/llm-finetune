# pip install datasets==2.14.7 fschat==0.2.34

import json

from datasets import load_dataset
from fastchat.model.model_adapter import get_conversation_template


def convert_using_fastchat(messages, model_name):
    context, reply = messages[:-1], messages[-1]
    assert reply["role"] == "assistant", f"Expected role assistant on last message, got {reply['role']}"
    conv = get_conversation_template(model_name)
    for message in context:
        msg_role = message["role"]
        if msg_role == "system":
            conv.system_message = message["content"]
        elif msg_role == "user":
            conv.append_message(conv.roles[0], message["content"])
        elif msg_role == "assistant":
            conv.append_message(conv.roles[1], message["content"])
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    conv.update_last_message(reply["content"])
    full = conv.get_prompt()
    completion = full[len(prompt) :].strip(" ")
    return {"prompt": prompt, "completion": completion}


def convert_and_write(model_name, data, split_names, filepath):
    rows = []
    for split_name in split_names:
        for row in data[split_name]:
            rows.append(convert_using_fastchat(row["messages"], model_name))
    with open(filepath, "w") as f:
        for row in rows:
            f.write(json.dumps(row))
            f.write("\n")


data = load_dataset("HuggingFaceH4/lima_llama2")
convert_and_write(
    model_name="NousResearch/Llama-2-7b-chat-hf", data=data, split_names=["train"], filepath="lima_llama2_1k.jsonl"
)
