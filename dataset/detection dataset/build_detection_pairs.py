import json
import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel, BertTokenizer, BertModel
from tqdm import tqdm
import time
from faiss_search import faiss_search


class CodeDataset(Dataset):
    def __init__(
        self,
        model,
        tokenizer,
        code_file_path=None,
        code_idx_max=100000,
        batch_size=32,
        device=None
    ):
        self.datas = []
        self.codes = []
        self.other_info = []

        if device is None:
            device = next(model.parameters()).device

        batch_codes = []

        def process_batch():
            if len(batch_codes) == 0:
                return

            tokens = tokenizer(
                batch_codes,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=256,
                add_special_tokens=True
            )

            tokens = {k: v.to(device) for k, v in tokens.items()}

            with torch.no_grad():
                outputs = model(**tokens)
                vecs = outputs[1]

            for i in range(vecs.size(0)):
                self.datas.append(vecs[i].cpu())

        with open(code_file_path, "r", encoding="utf-8") as f:
            idx = 0
            for line in f:
                idx += 1

                if idx % 1000 == 0:
                    print(idx)

                if idx > code_idx_max:
                    break

                code_info_json = json.loads(line.strip())

                code_tokens = code_info_json['code_tokens']
                code = " ".join(code_tokens)

                self.codes.append(code)
                self.other_info.append(code_info_json)

                batch_codes.append(code)

                if len(batch_codes) == batch_size:
                    process_batch()
                    batch_codes = []

            process_batch()

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, i):
        return self.datas[i]


def load_model(model_name, device):

    if model_name == "codebert-base":
        tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
        model = AutoModel.from_pretrained("microsoft/codebert-base")

    elif model_name == "bert":
        tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
        model = BertModel.from_pretrained("bert-base-cased")

    else:
        raise ValueError("Unknown model")

    model.to(device)
    model.eval()

    return tokenizer, model


def process(lang, dataset, args, tokenizer, model, device):

    saved_dir = f"../detection dataset/{lang}/{dataset}"
    os.makedirs(saved_dir, exist_ok=True)

    code_file_name = f"../detection dataset/pairs/{lang}/{dataset}.jsonl"

    idx_max = sum(1 for _ in open(code_file_name, 'r', encoding="utf-8"))
    print(f"{lang} {dataset} size: {idx_max}")

    pos_neg_result_file_name = f"{saved_dir}/top{args.topk}_similar.jsonl"
    codebase_embedding_file = f"{saved_dir}/{dataset}_{idx_max}.pth"

    if os.path.exists(codebase_embedding_file):
        print("loading cached embedding")
        code_dataset = torch.load(codebase_embedding_file)

    else:
        code_dataset = CodeDataset(
            model,
            tokenizer,
            code_file_name,
            idx_max,
            args.batch_size,
            device
        )
        torch.save(code_dataset, codebase_embedding_file)
        print("saved embedding")

    dataloader = DataLoader(code_dataset, batch_size=args.batch_size)

    code_vecs = []

    for batch in tqdm(dataloader, desc="concatenating"):
        code_vecs.append(batch.cpu().numpy())

    code_vecs = np.concatenate(code_vecs, 0)

    print("FAISS searching...")
    start = time.time()

    scores, sort_ids = faiss_search(code_vecs, args.topk, code_vecs)

    print(f"time: {(time.time()-start)/60:.2f} min")

    results = []

    for i in range(len(sort_ids)):

        pos_info = code_dataset.other_info[i]

        result_obj = {
            "code_pos": code_dataset.codes[i],
            "code_pos_doc": " ".join(pos_info['docstring_tokens']),
            "code_pos_func_name": pos_info['func_name']
        }

        for j in range(args.topk):

            neg_info = code_dataset.other_info[sort_ids[i][j]]

            result_obj[f"NO.{j+1}"] = " ".join(neg_info['code_tokens'])
            result_obj[f"score.{j+1}"] = float(scores[i][j])
            result_obj[f"func_name.{j+1}"] = neg_info['func_name']
            result_obj[f"doc.{j+1}"] = " ".join(neg_info['docstring_tokens'])

        results.append(result_obj)

    with open(pos_neg_result_file_name, "w", encoding="utf-8") as f:
        for r in results:
            json.dump(r, f)
            f.write("\n")

    print(f"saved: {pos_neg_result_file_name}")
    print("done")
    print("=============================================")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--language", default="all")
    parser.add_argument("--dataset", default="all")
    parser.add_argument("--model", default="codebert-base")
    parser.add_argument("--topk", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=64)

    args = parser.parse_args()

    ALL_LANGS = ["ruby", "javascript", "php", "go", "java", "python"]
    ALL_DATASETS = ["train", "valid", "test"]

    langs = ALL_LANGS if args.language == "all" else [args.language]
    datasets = ALL_DATASETS if args.dataset == "all" else [args.dataset]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    tokenizer, model = load_model(args.model, device)

    for lang in langs:
        for ds in datasets:
            process(lang, ds, args, tokenizer, model, device)