from model.HCLModel import HCLModel
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from tqdm import tqdm
from transformers import RobertaTokenizer, RobertaModel
import os
from torch.utils.data import Dataset
from functools import partial
import logging
import time
import argparse

class ContrastiveDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        text = []
        label = []
        self.sep_token = ['[sep]']
        self.label_list = ['[pos]', '[neg]']

        for i, data in enumerate(texts):
            tokens = data.lower().split(' ')
            label_id = labels[i]
            text.append(self.label_list + self.sep_token + tokens)
            label.append(label_id)
        self.text = text
        self.label = label

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = self.text[index]
        label = self.label[index]
        return text, label

def my_collate(batch, tokenizer, method, num_classes, args):
    tokens, label = map(list, zip(*batch))
    text_ids = tokenizer(tokens,
                          padding=True,
                          truncation=True,
                          max_length= (args.nl_length + args.code_length - 2),
                          is_split_into_words=True,
                          add_special_tokens=True,
                          return_tensors='pt')
    return text_ids, torch.tensor(label)

class hedgeLoss(nn.Module):
    def __init__(self, alpha, temp, loss_type):
        super().__init__()
        self.xent_loss = nn.CrossEntropyLoss()
        self.alpha = alpha
        self.temp = temp
        self.margin = 1.0
        self.loss_type = loss_type

    def forward(self, outputs, targets):
        if self.training and self.loss_type == 'hcl':
            anchor_cls_feats = self.normalize_feats(outputs['cls_feats'])
            anchor_label_feats = self.normalize_feats(outputs['label_feats'])
            neg_cls_feats = self.normalize_feats(outputs['neg_cls_feats'])
            pos_cls_feats = self.normalize_feats(outputs['pos_cls_feats'])
            pos_label_feats = self.normalize_feats(outputs['pos_label_feats'])

            normed_pos_label_feats = torch.gather(pos_label_feats, dim=1, index=targets['label'].reshape(-1, 1, 1).expand(-1, 1, pos_label_feats.size(-1))).squeeze(1)
            normed_anchor_label_feats = torch.gather(anchor_label_feats, dim=1, index=targets['label'].reshape(-1, 1, 1).expand(-1, 1, anchor_label_feats.size(-1))).squeeze(1)
            normed_neg_label_feats = torch.mul(normed_anchor_label_feats, outputs['gamms'].unsqueeze(1))

            ce_loss = (1 - self.alpha) * self.xent_loss(outputs['predicts'], targets['label'])
            cl_loss_1 = 0.5 * self.alpha * self.hedge_loss(anchor_cls_feats, normed_pos_label_feats, normed_neg_label_feats)  # data view
            cl_loss_2 = 0.5 * self.alpha * self.hedge_loss(normed_anchor_label_feats, pos_cls_feats, neg_cls_feats)  # classifier view
            return ce_loss + cl_loss_1 + cl_loss_2
        else:
            ce_loss = self.xent_loss(outputs['predicts'], targets['label'])
            return ce_loss

    def hedge_loss(self, anchor, positive, negative):
        sim_pos = F.cosine_similarity(anchor, positive, dim=-1)
        sim_neg = F.cosine_similarity(anchor, negative, dim=-1)

        logits = torch.stack([sim_pos, sim_neg], dim=1) / self.temp
        labels = torch.zeros(anchor.size(0), dtype=torch.long, device=anchor.device)

        return F.cross_entropy(logits, labels)

    def normalize_feats(self, _feats):
        return F.normalize(_feats, dim=-1)

def train(args, model, logger, train_loader, optimizer, criterion, device, valid_loader, saved_dir, use_amp=True, scaler=None):
    best_valid_loss = np.inf
    best_valid_accuracy = 0.0
    total_epochs = args.trained_epochs + args.num_train_epochs
    for epoch in range(args.trained_epochs, total_epochs):
        model.train()
        criterion.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        bar = tqdm(enumerate(train_loader))
        for batch_idx, (text, label) in bar:
            text = text.to(device)
            label = label.to(device)
            targets = {
                'label': label
            }
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(text)
                loss = criterion(outputs, targets)

            optimizer.zero_grad()
            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            logits1 = outputs['predicts']
            total_loss += loss.item()
            preds = torch.argmax(logits1, dim=1)
            correct = (preds == label).sum().item()
            total_correct += correct
            total_samples += label.size(0)

            if batch_idx % 500 == 0:
                batch_acc = (preds == label).float().mean().item()
                logger.info(
                    f'Epoch {epoch} - Batch {batch_idx}/{len(train_loader)} - Loss: {loss.item():.4f} - Train Accuracy: {batch_acc:.4f}')
                print(
                    f'Epoch {epoch} - Batch {batch_idx}/{len(train_loader)} - Loss: {loss.item():.4f} - Train Accuracy: {batch_acc:.4f}')

        train_loss = total_loss / len(train_loader)
        train_acc = total_correct / total_samples
        logger.info(f'Epoch {epoch} - Train loss: {train_loss:.4f} - Train accuracy: {train_acc:.4f}')

        if valid_loader is not None:
            valid_loss, valid_acc = validate(model, criterion, device, valid_loader)
            logger.info(f'Epoch {epoch} - Valid loss: {valid_loss:.4f} - Valid accuracy: {valid_acc:.4f}')
            if valid_acc > best_valid_accuracy:
                best_valid_accuracy = valid_acc
                best_valid_loss = valid_loss

    checkpoint_path = os.path.join(saved_dir, "detector.pth")

    torch.save({
        'epoch': total_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict() if scaler is not None else None,
    }, checkpoint_path)

    logger.info(f"Checkpoint saved at epoch {total_epochs}")

    logger.info('Training Finish !!!!!!!!')
    logger.info(f'best valid loss == {best_valid_loss}, best valid accuracy == {best_valid_accuracy}')

    return model

def validate(model, criterion, device, valid_loader):
    print("________________valid_______________")
    model.eval()
    criterion.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for text, label in valid_loader:
            text = text.to(device)
            label = label.to(device)
            targets = {
                'label': label
            }
            outputs = model(text)
            loss = criterion(outputs, targets)
            logits = outputs['predicts']
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct = (preds == label).sum().item()
            total_correct += correct
            total_samples += label.size(0)
    valid_loss = total_loss / len(valid_loader)
    valid_acc = total_correct / total_samples
    return valid_loss, valid_acc

def read_datasets(lang, logger, args):
    dataset_arr = ["test", "train", "valid"]

    train_texts, test_texts, valid_texts = [], [], []
    train_labels, test_labels, valid_labels = [], [], []

    for dataset in dataset_arr:
        dataset_file_name = f"{args.detection_dir}/{lang}/{dataset}/{dataset}.h5"

        if not os.path.exists(dataset_file_name):
            logger.warning(f"{dataset_file_name} NOT FOUND → use empty dataset")
            texts, labels = [], []
        else:
            dt = torch.load(dataset_file_name)

            posinput = dt['pos_pairs']
            neginput = dt['neg_pairs']
            poslabels = dt['pos_labels']
            neglabels = dt['neg_labels']

            texts = posinput + neginput
            labels = poslabels + neglabels

        if len(texts) == 0:
            # keep empty
            _texts = np.array([])
            _labels = np.array([])
        else:
            _texts = np.array(texts)
            _labels = np.array(labels)

            perm = np.random.permutation(len(_texts))
            _texts = _texts[perm]
            _labels = _labels[perm]

        logger.info(f"{dataset} dataset length: {len(_texts)}")

        if dataset == "train":
            train_texts, train_labels = _texts, _labels
        elif dataset == "test":
            test_texts, test_labels = _texts, _labels
        elif dataset == "valid":
            valid_texts, valid_labels = _texts, _labels

    return train_texts, test_texts, valid_texts, train_labels, test_labels, valid_labels

if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    parser = argparse.ArgumentParser()

    parser.add_argument("--language", default=None, type=str, required=True, help="The programming language.")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--detection_dir", default=None, type=str, required=True,
                        help="The folder of detection pair datasets.")
    parser.add_argument('--encoder', type=str, default='codebert', choices=['codebert', 'unixcoder', 'cocosoda'])
    parser.add_argument("--nl_length", default=128, type=int,
                        help="Optional NL input sequence length after tokenization.")
    parser.add_argument("--code_length", default=256, type=int,
                        help="Optional Code input sequence length after tokenization.")
    parser.add_argument('--loss_type', type=str, default='ce', choices=['ce', 'hcl'], help="Loss function type.")
    parser.add_argument("--batch_size", default=4, type=int, help="Batch size.")
    parser.add_argument("--learning_rate", default=1e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=1, type=int, help="Total number of training epochs to perform.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--trained_epochs", default=0, type=int,
                    help="Number of epochs already trained (for resuming).")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lr = args.learning_rate
    epochs = args.num_train_epochs
    batch_size = args.batch_size
    loss_type = args.loss_type
    lang = args.language
    output_dir = args.output_dir
    encoder_name = args.encoder
    alpha = 0.5
    temp = 0.1

    total_epochs = args.trained_epochs + args.num_train_epochs
    base_dir = f"{output_dir}/{lang}/{encoder_name}/{loss_type}"
    saved_dir = f"{base_dir}/{total_epochs}_epochs"

    if not os.path.exists(saved_dir):
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

    if encoder_name == 'cocosoda':
        tokenizer = RobertaTokenizer.from_pretrained("DeepSoftwareAnalytics/CoCoSoDa")
        encoder = RobertaModel.from_pretrained(f"DeepSoftwareAnalytics/CoCoSoDa")
    if encoder_name == 'unixcoder':
        tokenizer = RobertaTokenizer.from_pretrained("microsoft/unixcoder-base")
        encoder = RobertaModel.from_pretrained(f"microsoft/unixcoder-base")
    if encoder_name == 'codebert':
        tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
        encoder = RobertaModel.from_pretrained(f"microsoft/codebert-base")
    special_tokens = {
        "additional_special_tokens": ['[POS]', '[NEG]']
    }

    tokenizer.add_special_tokens(special_tokens)

    hyper_parameter = f"hyper-parameter: - lr: {lr}; epochs: {epochs}; batch_size: {batch_size}"
    logger.info(hyper_parameter)
    logger.info(f"encoder_name: {encoder_name}")
    hidden_size = encoder.config.hidden_size

    encoder.train()
    model = HCLModel(encoder, args=args, tokenizer=tokenizer, hidden_size=hidden_size).to(device)
    for name, param in model.encoder.named_parameters():
        if "encoder.layer.10" in name or "encoder.layer.11" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    model.encoder.resize_token_embeddings(len(tokenizer))

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05, betas=(0.9, 0.99), eps=1e-8, amsgrad=True)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)
    
    if args.trained_epochs > 0:
        prev_dir = f"{base_dir}/{args.trained_epochs}_epochs"
        prev_ckpt_path = os.path.join(prev_dir, "detector.pth")

        if not os.path.exists(prev_ckpt_path):
            raise FileNotFoundError(
                f"Checkpoint not found at {prev_ckpt_path}"
            )

        logger.info(f"Loading checkpoint from {prev_ckpt_path}")
        checkpoint = torch.load(prev_ckpt_path, map_location=device)

        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scaler is not None and checkpoint['scaler_state_dict'] is not None:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])

        logger.info(f"Resumed from epoch {args.trained_epochs} → will train for {epochs} more epochs (total {total_epochs} epochs)")
    else:
        logger.info("trained_epochs = 0 → initialize from pretrained encoder.")
    
    logger.info(f"model structure: ")
    logger.info(f"=======================================================================================")
    logger.info(model)
    logger.info(f"=======================================================================================")

    
    train_texts, test_texts, val_texts, train_labels, test_labels, val_labels = read_datasets(lang, logger, args)
    train_dataset = ContrastiveDataset(train_texts, train_labels, tokenizer)
    val_dataset = ContrastiveDataset(val_texts, val_labels, tokenizer)
    test_dataset = ContrastiveDataset(test_texts, test_labels, tokenizer)

    collate_fn = partial(my_collate, tokenizer=tokenizer, method=None, num_classes=2, args=args)
    criterion = hedgeLoss(alpha, temp, loss_type)

    if args.do_train:
        # training
        train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=4, pin_memory=True, persistent_workers=True)
        val_loader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=4, pin_memory=True, persistent_workers=True)
        train_date = ''.join(str(datetime.now().date()).split("-"))
        model = train(args, model, logger, train_loader, optimizer, criterion, device, valid_loader=val_loader, saved_dir=saved_dir, use_amp=use_amp, scaler=scaler)

    if args.do_eval:
        # evaluation
        test_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        encoder.eval()
        model.eval()
        criterion.eval()
        total_acc = 0
        y_true = []
        y_pred = []
        with torch.no_grad():
            for text, label in test_loader:
                text = text.to(device)
                label = label.to(device)

                outputs = model(text)
                logits = outputs['predicts']
                y_true.append(label)
                y_pred.append(torch.argmax(logits, -1))
                preds = torch.argmax(logits, dim=1)
                correct = (preds == label).sum().item()
                total_acc += correct / label.size(0)

        test_acc = total_acc / len(test_loader)

        logger.info(f'Test accuracy: {test_acc:.4f}')