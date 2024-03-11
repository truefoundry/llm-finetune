import os


def find_all_jsonl_files(path):
    for root, dirs, files in os.walk(path):
        for file in files:
            filepath = os.path.join(root, file)
            filename = os.path.basename(filepath)
            if filename.endswith(".jsonl") and not filename.startswith("."):
                yield filepath
