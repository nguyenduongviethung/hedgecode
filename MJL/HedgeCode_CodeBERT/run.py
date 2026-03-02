import argparse
import logging
import os
import random
import time
import torch
import json
import numpy as np
from torch.utils.data import SubsetRandomSampler
from model.HCLModel import HCLModel
from model.MJLModel import MJLModel
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler
from transformers import get_linear_schedule_with_warmup, RobertaConfig, RobertaModel, RobertaTokenizer
from torch.optim import AdamW


def read_datasets(lang, logger, args):
    dataset_arr = ["train", "valid", "test", "codebase"]

    train_texts, valid_texts, test_texts, codebase_texts = [], [], [], []

    for dataset in dataset_arr:
        dataset_file_path = f"{args.dataset_dir}/{lang}/{dataset}.jsonl"

        data = []
        with open(dataset_file_path) as f:
            for line in f:
                line = line.strip()
                js = json.loads(line)
                data.append(js)
        
        logger.info(f"{dataset} dataset length: {len(data)}")

        if dataset == "train":
            train_texts = data
        elif dataset == "valid":
            valid_texts = data
        elif dataset == "test":
            test_texts = data
        elif dataset == "codebase":
            codebase_texts = data

    return train_texts, valid_texts, test_texts, codebase_texts

class InputFeatures(object):
    """A single training/test features for a example."""

    def __init__(self,
                 code_tokens,
                 code_ids,
                 nl_tokens,
                 nl_ids,
                 url
                 ):
        self.code_tokens = code_tokens
        self.code_ids = code_ids
        self.nl_tokens = nl_tokens
        self.nl_ids = nl_ids
        self.url = url

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, args):
        self.examples = []
        for js in texts:
            code = ' '.join(js['code_tokens'])
            code_tokens = tokenizer.tokenize(code)[:args.code_length - 2]
            code_tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.sep_token]
            code_ids = tokenizer.convert_tokens_to_ids(code_tokens)
            padding_length = args.code_length - len(code_ids)
            code_ids += [tokenizer.pad_token_id] * padding_length

            nl = ' '.join(js['docstring_tokens'])
            nl_tokens = tokenizer.tokenize(nl)[:args.nl_length - 2]
            nl_tokens = [tokenizer.cls_token] + nl_tokens + [tokenizer.sep_token]
            nl_ids = tokenizer.convert_tokens_to_ids(nl_tokens)
            padding_length = args.nl_length - len(nl_ids)
            nl_ids += [tokenizer.pad_token_id] * padding_length
            self.examples.append(InputFeatures(code_tokens, code_ids, nl_tokens, nl_ids, js['url']))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return (torch.tensor(self.examples[i].code_ids), torch.tensor(self.examples[i].nl_ids))


def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def train(args, model, logger, optimizer, valid_dataset, codebase_dataset, train_dataloader, valid_dataloader, codebase_dataloader, saved_dir, use_amp=True, scaler=None):
    total_epochs = args.trained_epochs + args.num_train_epochs
    best_mrr = 0.0

    for epoch in range(args.trained_epochs, total_epochs):
        model.train()
        total_loss = 0

        for step, batch in enumerate(train_dataloader):
            code_inputs = batch[0].to(args.device)
            nl_inputs = batch[1].to(args.device)

            with torch.amp.autocast(
                device_type=args.device.type,
                enabled=(args.device.type == "cuda")
            ):
                loss = model(code_inputs=code_inputs, nl_inputs=nl_inputs)

                if loss.dim() > 0:
                    loss = loss.mean()

            optimizer.zero_grad()

            if args.device.type == "cuda":
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            total_loss += loss.item()

            if step % 500 == 0:
                logger.info(
                    f"Epoch {epoch} - Step {step}/{len(train_dataloader)} "
                    f"- Loss: {loss.item():.4f}"
                )

        logger.info(
            f"Epoch {epoch} - Train loss: "
            f"{total_loss / len(train_dataloader):.4f}"
        )

        results = evaluate(
            args, model, valid_dataset, codebase_dataset, valid_dataloader, codebase_dataloader,
            eval_when_training=True
        )

        logger.info(f"Epoch {epoch} - Eval MRR: {results['eval_mrr']:.4f}")

        if results['eval_mrr'] > best_mrr:
            best_mrr = results['eval_mrr']

    checkpoint_path = os.path.join(saved_dir, "detector.pth")

    torch.save({
        'epoch': total_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict() if scaler else None,
    }, checkpoint_path)

    logger.info(f"Checkpoint saved at epoch {total_epochs}")


def evaluate(args, model, query_dataset, code_dataset, query_dataloader, code_dataloader, eval_when_training=False):
    model.eval()
    code_vecs = []
    nl_vecs = []
    for batch in query_dataloader:
        nl_inputs = batch[1].to(args.device)
        with torch.no_grad():
            nl_vec = model(nl_inputs=nl_inputs)
            nl_vecs.append(nl_vec.cpu().numpy())

    for batch in code_dataloader:
        code_inputs = batch[0].to(args.device)
        with torch.no_grad():
            code_vec = model(code_inputs=code_inputs)
            code_vecs.append(code_vec.cpu().numpy())

    code_vecs = np.concatenate(code_vecs, 0)
    nl_vecs = np.concatenate(nl_vecs, 0)

    scores = np.matmul(nl_vecs, code_vecs.T)

    sort_ids = np.argsort(scores, axis=-1, kind='quicksort', order=None)[:, ::-1]

    nl_urls = []
    code_urls = []
    for example in query_dataset.examples:
        nl_urls.append(example.url)

    for example in code_dataset.examples:
        code_urls.append(example.url)

    ranks = []
    for url, sort_id in zip(nl_urls, sort_ids):
        rank = 0
        find = False
        for idx in sort_id[:1000]:
            if find is False:
                rank += 1
            if code_urls[idx] == url:
                find = True
        if find:
            ranks.append(1 / rank)
        else:
            ranks.append(0)

    result = {
        "eval_mrr": float(np.mean(ranks))
    }
    if eval_when_training:
        model.train()

    return result


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--dataset_dir", default=None, type=str, required=True,
                        help="The input dataset directory which contains train.jsonl, valid.jsonl, test.jsonl and codebase.jsonl.")
    parser.add_argument("--detector_dir", type=str, default=None,
                        help="Directory containing detector.pth")
    
    parser.add_argument("--nl_length", default=128, type=int,
                        help="Optional NL input sequence length after tokenization.")
    parser.add_argument("--code_length", default=256, type=int,
                        help="Optional Code input sequence length after tokenization.")

    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_valid", action='store_true',
                        help="Whether to run eval on the valid set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the test set.")

    parser.add_argument("--fewshot", default=False, action='store_true', required=False, help="do shot setting")

    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="Batch size for training.")
    parser.add_argument("--eval_batch_size", default=4, type=int,
                        help="Batch size for evaluation.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=1, type=int,
                        help="Total number of training epochs to perform.")

    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument("--language", type=str, required=True)
    parser.add_argument("--encoder", type=str, default="codebert",
                        choices=["codebert", "unixcoder", "cocosoda"])
    parser.add_argument("--trained_epochs", default=0, type=int,
                        help="Number of epochs already trained (for resuming).")


    # arguments
    args = parser.parse_args()

    total_epochs = args.trained_epochs + args.num_train_epochs
    base_dir = f"{args.output_dir}/{args.language}/{args.encoder}"
    saved_dir = f"{base_dir}/{total_epochs}_epochs"
    os.makedirs(saved_dir, exist_ok=True)
    
    current_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    logger = logging.getLogger('train_logger')
    logger.setLevel(logging.INFO)

    log_file_name = os.path.join(saved_dir, f"{current_time}.log")
    with open(log_file_name, 'w') as file:
        pass
    file_handler = logging.FileHandler(log_file_name)
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    logger.info("device: %s, n_gpu: %s", device, args.n_gpu)

    # Set seed
    set_seed(args.seed)

    # build model
    if args.encoder == 'cocosoda':
        tokenizer = RobertaTokenizer.from_pretrained("DeepSoftwareAnalytics/CoCoSoDa")
        encoder = RobertaModel.from_pretrained("DeepSoftwareAnalytics/CoCoSoDa")
    if args.encoder == 'unixcoder':
        tokenizer = RobertaTokenizer.from_pretrained("microsoft/unixcoder-base")
        encoder = RobertaModel.from_pretrained("microsoft/unixcoder-base")
    if args.encoder == 'codebert':
        tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
        encoder = RobertaModel.from_pretrained("microsoft/codebert-base")
    special_tokens = {
        "additional_special_tokens": ['[POS]', '[NEG]']
    }
    tokenizer.add_special_tokens(special_tokens)
    encoder.resize_token_embeddings(len(tokenizer))

    hidden_size = encoder.config.hidden_size

    # ========== BUILD MODEL FIRST ==========
    if args.trained_epochs > 0:

        prev_dir = f"{base_dir}/{args.trained_epochs}_epochs"
        prev_ckpt_path = os.path.join(prev_dir, "detector.pth")

        if not os.path.exists(prev_ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found at {prev_ckpt_path}")

        logger.info(f"Loading checkpoint from {prev_ckpt_path}")
        checkpoint = torch.load(prev_ckpt_path, map_location=args.device)

        model = MJLModel(encoder, tokenizer, args).to(args.device)
        model.load_state_dict(checkpoint['model_state_dict'])

    else:

        if args.detector_dir is not None:
            detector_path = os.path.join(args.detector_dir, "detector.pth")
            logger.info(f"Loading detector from {detector_path}")

            checkpoint = torch.load(detector_path, map_location=args.device)

            detector = HCLModel(
                encoder,
                args=args,
                tokenizer=tokenizer,
                hidden_size=hidden_size
            ).to(args.device)

            detector.load_state_dict(checkpoint['model_state_dict'], strict=False)

            encoder = detector.encoder

        model = MJLModel(encoder, tokenizer, args).to(args.device)

    # ========== NOW BUILD OPTIMIZER ==========
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=0.05,
        betas=(0.9, 0.99),
        eps=1e-8,
        amsgrad=True
    )

    use_amp = args.device.type == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)

    # ========== LOAD OPTIMIZER STATE IF RESUME ==========
    if args.trained_epochs > 0:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if scaler is not None and checkpoint['scaler_state_dict'] is not None:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])

        logger.info(
            f"Resumed from epoch {args.trained_epochs} → "
            f"will train {args.num_train_epochs} more epochs"
        )

    logger.info(f"model structure: ")
    logger.info(f"=======================================================================================")
    logger.info(model)
    logger.info(f"=======================================================================================")

    train_texts, valid_texts, test_texts, codebase_texts = read_datasets(args.language, logger, args)
    train_dataset = TextDataset(train_texts, tokenizer, args)
    valid_dataset = TextDataset(valid_texts, tokenizer, args)
    test_dataset = TextDataset(test_texts, tokenizer, args)
    codebase_dataset = TextDataset(codebase_texts, tokenizer, args)

    train_sampler = RandomSampler(train_dataset)

    if args.fewshot:
        logger.info("Doing few-shot training")
        all_indices = list(range(len(train_dataset)))
        random_indices = random.sample(all_indices, int(len(train_dataset) * 0.2))
        train_sampler = SubsetRandomSampler(random_indices)

    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, num_workers=4)

    valid_sampler = SequentialSampler(valid_dataset)
    valid_dataloader = DataLoader(valid_dataset, sampler=valid_sampler, batch_size=args.eval_batch_size, num_workers=4)

    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.eval_batch_size, num_workers=4)

    codebase_sampler = SequentialSampler(codebase_dataset)
    codebase_dataloader = DataLoader(codebase_dataset, sampler=codebase_sampler, batch_size=args.eval_batch_size, num_workers=4)

    # Training
    if args.do_train:
        train(args, model, logger, optimizer, valid_dataset, codebase_dataset, train_dataloader, valid_dataloader, codebase_dataloader, saved_dir, use_amp, scaler)

    # Evaluation
    results = {}
    if args.do_valid:
        result = evaluate(args, model, valid_dataset, codebase_dataset, valid_dataloader, codebase_dataloader)
        logger.info("***** Valid results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(round(result[key], 4)))

    if args.do_test:
        result = evaluate(args, model, test_dataset, codebase_dataset, test_dataloader, codebase_dataloader)
        logger.info("***** Test results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(round(result[key], 4)))

    return results


if __name__ == "__main__":
    main()