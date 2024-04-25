import numpy as np
import torch
import argparse
import os
import random

from transformers import AutoModel, AutoTokenizer
from transformers import PreTrainedTokenizer, BertConfig, BertForTokenClassification
from transformers import AdamW
from tqdm import tqdm, trange
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report
from torch.utils.data import DataLoader, Dataset
from libraries_ner import load_data, NerDataset
from dataclasses import dataclass


def define_argparser():
    '''process the arguments in the commandline'''
    p = argparse.ArgumentParser()
    p.add_argument('--model_fn', dest='path_out_model', type=str, required=True)
    p.add_argument('--train_fn', dest='train_data', type=str, required=True)
    p.add_argument('--pretrained_model_name', dest='model_name', type=str, default="dmis-lab/biobert-v1.1")
    p.add_argument('--valid_ratio', type=float, default=0.2)
    p.add_argument('--batch_size_per_device', dest='batch_size', type=int, default=16)
    p.add_argument('--n_epochs', dest='epoch', type=int, default=30)
    p.add_argument('--max_seq_len', type=int, default=400)
    p.add_argument('--learning_rate', type=float, default=5e-3)
    p.add_argument('--adam_epsilon', type=float, default=1e-8)
    p.add_argument('--device', type=str, default="cuda")
    p.add_argument('--max_grad_norm', type=float, default=1.0)
    p.add_argument('--seed', type=int, default=1234)
    config = p.parse_args()

    return config


def collate_fn(input_examples):
    '''collate tokenized dataset'''
    input_texts, input_labels_str = [], []
    offset_mappings = []

    for input_example in input_examples:
        text, label_strs = input_example["sentence"], input_example["token_label"]
        input_texts.append(text)
        input_labels_str.append(label_strs)
        offset_mappings.append(input_example["offset_mapping"])

    encoded_texts = tokenizer.batch_encode_plus(
        input_texts,
        add_special_tokens=True,
        max_length=max_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
        return_token_type_ids=True,
        return_attention_mask=True,
        return_offsets_mapping=True
    )
    
    input_ids = encoded_texts["input_ids"]
    token_type_ids = encoded_texts["token_type_ids"]
    attention_mask = encoded_texts["attention_mask"]

    len_input = input_ids.size(1)
    input_labels = []
    for input_label_str in input_labels_str:
        input_label = [label2id[x] for x in input_label_str]
        if len(input_label) > max_length - 2:
            input_label = input_label[:max_length - 2]
            input_label = [-100] + input_label + [-100]
        else:
            input_label = (
                [-100] + input_label + (max_length - len(input_label_str) - 1) * [-100]
            )
        input_label = torch.tensor(input_label).long()
        input_labels.append(input_label)

    input_labels = torch.stack(input_labels)
    
    return input_ids, token_type_ids, attention_mask, input_labels, offset_mappings


def train_epoch(dataloader, model, optimizer):
    model.train()
    total_loss = 0.0

    tepoch = tqdm(dataloader, unit="batch", position=1, leave=True)
    for batch in tepoch:
        tepoch.set_description(f"Train")
        model.zero_grad()

        input_ids = batch[0].to(config.device)
        token_type_ids = batch[1].to(config.device)
        attention_mask = batch[2].to(config.device)
        labels = batch[3].to(config.device)

        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "labels": labels,
        }

        outputs = model(**inputs)

        loss = outputs[0]
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
        optimizer.step()
        total_loss += loss.item()

        tepoch.set_postfix(loss=loss.mean().item())
    tepoch.set_postfix(loss=total_loss / len(dataloader))
    
    return total_loss / len(dataloader)


def valid_epoch(dataloader, model):
    model.eval()
    total_loss = 0.0

    all_token_predictions = []
    all_token_labels = []

    tepoch = tqdm(dataloader, unit="batch", leave=False)
    for batch in tepoch:
        tepoch.set_description(f"Valid")
        with torch.no_grad():
            input_ids = batch[0].to(config.device)
            token_type_ids = batch[1].to(config.device)
            attention_mask = batch[2].to(config.device)
            labels = batch[3].to(config.device)
            offset_mappings = batch[4]
            inputs = {
                "input_ids": input_ids,
                "token_type_ids": token_type_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            }

            outputs = model(**inputs)

            loss, logits = outputs[:2]
            total_loss += loss.item()

            token_predictions = logits.argmax(dim=2)  # logits
            token_predictions = token_predictions.detach().cpu().numpy()

            for token_prediction, label in zip(token_predictions, labels):
                filtered = []
                filtered_label = []
                for i in range(len(token_prediction)):
                    if label[i].tolist() == -100:
                        continue
                    filtered.append(id2label[token_prediction[i]])
                    filtered_label.append(id2label[label[i].tolist()])
                assert len(filtered) == len(filtered_label)
                all_token_predictions.append(filtered)
                all_token_labels.append(filtered_label)

        tepoch.set_postfix(loss=loss.mean().item())

    token_f1 = f1_score(all_token_labels, all_token_predictions, average="macro")

    return total_loss / len(dataloader), token_f1


init_config = define_argparser()

if os.path.exists(init_config.path_out_model):
    print("The model already exist. Stop")
    exit(0)

labels = ["B-Disease", "I-Disease", "B-Formula", "I-Formula", "O"]
label2id = {label: i for i, label in enumerate(labels)}
id2label = {i: label for label, i in label2id.items()}

tokenizer = AutoTokenizer.from_pretrained(init_config.model_name)

max_length = init_config.max_seq_len
batch_size = init_config.batch_size

torch.manual_seed(init_config.seed)
np.random.seed(init_config.seed)

config = BertConfig.from_pretrained(init_config.model_name, num_labels=len(label2id))
config.update(init_config.__dict__)

model = BertForTokenClassification.from_pretrained(config.model_name, config=config)
model.cuda()

dataset_train_vals = load_data(init_config.train_data, tokenizer)
dic_dataset_train_vals = {}

for tmp_test in dataset_train_vals:
    if tmp_test['pub_id'] not in dic_dataset_train_vals:
        dic_dataset_train_vals[tmp_test['pub_id']] = []
    dic_dataset_train_vals[tmp_test['pub_id']].append(tmp_test)

ids_dataset_train_vals = list(dic_dataset_train_vals.keys())
valid_ratio = init_config.valid_ratio

random.seed(init_config.seed)
random.shuffle(ids_dataset_train_vals)

split_index = int(len(ids_dataset_train_vals) * valid_ratio)
ids_validation = ids_dataset_train_vals[:split_index]
ids_train = ids_dataset_train_vals[split_index:]

dataset_train = []
dataset_validation = []

for tmp_key in ids_train:
    for tmp_data in dic_dataset_train_vals[tmp_key]:
        dataset_train.append(tmp_data)

for tmp_key in ids_validation:
    for tmp_data in dic_dataset_train_vals[tmp_key]:
        dataset_validation.append(tmp_data)

train_dataset = NerDataset(
    tokenizer=tokenizer,
    examples=dataset_train,
    max_length=max_length
)

train_dataloader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_fn
)

valid_dataset = NerDataset(
    tokenizer=tokenizer,
    examples=dataset_validation,
    max_length=max_length
)

valid_dataloader = DataLoader(
    dataset=valid_dataset,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=collate_fn
)

optimizer_grouped_parameters = [
    {'params': model.bert.parameters(), 'lr': config.learning_rate / 100 },
    {'params': model.classifier.parameters(), 'lr': config.learning_rate }
]
optimizer = AdamW(optimizer_grouped_parameters, lr=config.learning_rate, eps=config.adam_epsilon)

model.to(init_config.device)

best_f1 = 0.0
best_model = None

tepoch = trange(config.epoch, position=0, leave=True)
for epoch in tepoch:
    tepoch.set_description(f"Epoch {epoch}")

    train_loss = train_epoch(train_dataloader, model, optimizer)
    valid_loss, token_f1 = valid_epoch(valid_dataloader, model)

    if best_f1 < token_f1:
        best_f1 = token_f1
        best_model = model

    tepoch.set_postfix(valid_f1=token_f1)

torch.save({
    'config': init_config,
    'model': best_model.state_dict()
}, init_config.path_out_model)