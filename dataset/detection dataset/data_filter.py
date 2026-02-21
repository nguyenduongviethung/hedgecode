import json
import os
import argparse


def process(lang, dataset, args):

    print("================================================")
    print(f"processing: {lang} / {dataset}")

    pos_neg_file = f"../detection dataset/{lang}/{dataset}/top{args.topN}_similar.jsonl"

    dataset_data = []

    with open(pos_neg_file, 'r', encoding="utf-8") as f:
        for line in f:
            dataset_data.append(json.loads(line))

    print(f"source dataset length: {len(dataset_data)}")

    results = []

    score_key = f"score.{args.score_index}"

    for data in dataset_data:

        if score_key not in data:
            continue

        if data[score_key] <= args.threshold:

            filter_data = {
                "code_pos": data["code_pos"],
                "code_pos_doc": data["code_pos_doc"],
                "code_neg_1": data[f"NO.{args.neg_index_1}"],
                "code_neg_doc_1": data[f"doc.{args.neg_index_1}"],
                "code_neg_2": data[f"NO.{args.neg_index_2}"],
                "code_neg_doc_2": data[f"doc.{args.neg_index_2}"],
            }

            results.append(filter_data)

    print(f"filter dataset length: {len(results)}")

    saved_dir = f"../detection dataset/{lang}/{dataset}"
    os.makedirs(saved_dir, exist_ok=True)

    dataset_file_name = f"{saved_dir}/{dataset}_top{args.topK}.jsonl"

    with open(dataset_file_name, 'w', encoding="utf-8") as f:
        for d in results:
            json.dump(d, f)
            f.write('\n')

    print(f"saved: {dataset_file_name}")
    print("done")
    print("================================================")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--language", default="all")
    parser.add_argument("--dataset", default="all")

    parser.add_argument("--topN", default="4")
    parser.add_argument("--topK", default="2")

    parser.add_argument("--threshold", type=float, default=20.0)

    # which score index to filter on (default = 3)
    parser.add_argument("--score_index", type=int, default=3)

    # which negatives to pick
    parser.add_argument("--neg_index_1", type=int, default=2)
    parser.add_argument("--neg_index_2", type=int, default=3)

    args = parser.parse_args()

    ALL_LANGS = ["ruby", "javascript", "php", "go", "java", "python"]
    ALL_DATASETS = ["train", "valid", "test"]

    langs = ALL_LANGS if args.language == "all" else [args.language]
    datasets = ALL_DATASETS if args.dataset == "all" else [args.dataset]

    for lang in langs:
        for ds in datasets:
            process(lang, ds, args)