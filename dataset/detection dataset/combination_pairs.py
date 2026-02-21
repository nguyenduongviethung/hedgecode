import json
import os
import argparse
import torch


def process(lang, dataset, args):

    print("==============================================")
    print(f"processing: {lang} / {dataset}")

    dataset_file_name = f"../detection dataset/{lang}/{dataset}/{dataset}_top{args.topk}.jsonl"
    print(f"dataset file: {dataset_file_name}")

    pos_pairs = []
    neg_pairs = []

    sep_token = args.sep_token

    with open(dataset_file_name, 'r', encoding="utf-8") as f:

        for line in f:

            json_obj = json.loads(line)

            description = json_obj["code_pos_doc"]
            pos_code = json_obj["code_pos"]

            # negative candidates
            for k in range(1, args.topk + 1):

                neg_key = f"code_neg_{k}"

                # backward compatibility
                if neg_key not in json_obj:
                    neg_key = f"NO.{k}"

                neg_code = json_obj[neg_key]

                neg_pair = description + sep_token + neg_code
                neg_pairs.append(neg_pair)

            pos_pair = description + sep_token + pos_code
            pos_pairs.append(pos_pair)

    print(f"number of pos pairs: {len(pos_pairs)}")
    print(f"number of neg pairs: {len(neg_pairs)}")

    pos_labels = [1] * len(pos_pairs)
    neg_labels = [0] * len(neg_pairs)

    datadict = {
        "pos_pairs": pos_pairs,
        "neg_pairs": neg_pairs,
        "pos_labels": pos_labels,
        "neg_labels": neg_labels
    }

    print(f"total pairs: {len(pos_pairs) + len(neg_pairs)}")

    saved_pair_dir = f"../detection dataset/{lang}/{dataset}"
    os.makedirs(saved_pair_dir, exist_ok=True)

    save_path = f"{saved_pair_dir}/{dataset}.h5"

    torch.save(datadict, save_path)

    print(f"saved: {save_path}")
    print("done")
    print("==============================================")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--language", default="all")
    parser.add_argument("--dataset", default="all")
    parser.add_argument("--topk", type=int, default=2)
    parser.add_argument("--sep_token", default=" [SEP] ")

    args = parser.parse_args()

    ALL_LANGS = ["ruby", "javascript", "php", "go", "java", "python"]
    ALL_DATASETS = ["train", "valid", "test"]

    langs = ALL_LANGS if args.language == "all" else [args.language]
    datasets = ALL_DATASETS if args.dataset == "all" else [args.dataset]

    for lang in langs:
        for ds in datasets:
            process(lang, ds, args)