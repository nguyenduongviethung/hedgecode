import argparse
import json
import os
from datasets import load_dataset

ALL_LANGUAGES = ["python", "java", "go", "javascript", "php", "ruby"]

def convert_sample(sample):
    return {
        "url": sample["func_code_url"],
        "code": sample["func_code_string"],
        "code_tokens": sample["func_code_tokens"],
        "docstring": sample["func_documentation_string"],
        "docstring_tokens": sample["func_documentation_tokens"],
        "original_string": sample["func_code_string"],
    }

def process_language(lang):
    print("Processing", lang)

    ds = load_dataset("code_search_net", lang, trust_remote_code=True)

    os.makedirs(lang, exist_ok=True)

    with open(f"{lang}/train.jsonl", "w") as train_file, \
         open(f"{lang}/valid.jsonl", "w") as valid_file, \
         open(f"{lang}/test.jsonl", "w") as test_file, \
         open(f"{lang}/codebase.jsonl", "w") as codebase_file:

        # ---------- train ----------
        for sample in ds["train"]:
            js = convert_sample(sample)
            train_file.write(json.dumps(js) + "\n")

        # ---------- valid ----------
        for sample in ds["validation"]:
            js = convert_sample(sample)
            js["code"] = ""
            js["code_tokens"] = []
            js["original_string"] = ""
            valid_file.write(json.dumps(js) + "\n")

            cb = convert_sample(sample)
            cb["docstring"] = ""
            cb["docstring_tokens"] = []
            codebase_file.write(json.dumps(cb) + "\n")

        # ---------- test ----------
        for sample in ds["test"]:
            js = convert_sample(sample)
            js["code"] = ""
            js["code_tokens"] = []
            js["original_string"] = ""
            test_file.write(json.dumps(js) + "\n")

            cb = convert_sample(sample)
            cb["docstring"] = ""
            cb["docstring_tokens"] = []
            codebase_file.write(json.dumps(cb) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", type=str, default="all",
                        help="Language to preprocess")

    args = parser.parse_args()

    if args.language == "all":
        languages = ALL_LANGUAGES
    else:
        languages = [args.language]

    for lang in languages:
        process_language(lang)

    print("Done.")