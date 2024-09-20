import sys
from transformers import AutoModel, AutoTokenizer
import numpy as np
import torch
import copy
import argparse
import os

from transformers import PreTrainedTokenizer, BertConfig, BertForTokenClassification, ElectraForTokenClassification, DebertaV2ForTokenClassification, DebertaForTokenClassification
from tqdm import tqdm, trange
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report
from dataclasses import dataclass
from torch.utils.data import DataLoader, Dataset
from libraries_ner import load_data, NerDataset


def define_argparser():
    '''process the arguments in the commandline'''
    p = argparse.ArgumentParser()
    p.add_argument('--model_fn', dest='path_model', required=True)
    p.add_argument('--eval_fn', dest='test_data', required=True)
    p.add_argument('--output_fn', required=True)
    p.add_argument('--device', type=str, default="cuda")
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--max_seq_len', type=int, default=400)
    p.add_argument('--seed', type=int, default=1234)
    config = p.parse_args()

    return config


def token_pred_to_char_pred(token_prediction, token_label, offset_mapping):
    '''map predicted tokens to charaters prediction'''
    offset_mapping = copy.deepcopy(offset_mapping)
    char_predictions = []
    filtered = []

    for i in range(len(token_prediction)):
        if token_label[i].tolist() == -100:
            continue
        filtered.append(token_prediction[i])
    char_prediction = []

    if offset_mapping[0][0] == 0 and offset_mapping[0][1] == 0:
        del offset_mapping[0]
    if offset_mapping[-1][0] == 0 and offset_mapping[-1][1] == 0:
        del offset_mapping[-1]
    assert len(filtered) == len(offset_mapping)

    prev_end = None
    for token_predict, offset_mapping in zip(filtered, offset_mapping):
        start, end = offset_mapping

        if prev_end != None and start - prev_end > 0:
            char_prediction.append("O")
        prev_end = end

        if end - start == 1:
            label_str = id2label[token_predict]
            char_prediction.append(label_str)
            continue

        for i in range(end - start):
            label_str = id2label[token_predict]
            if i == 0 or label_str == "O":
                char_prediction.append(label_str)
                continue
            char_prediction.append("I-" + label_str.split("-")[1])

    return char_prediction


def collate_fn(input_examples):
    '''collate tokenized dataset'''
    input_texts, input_labels_str = [], []
    offset_mappings = []
    char_labels = []
    pub_ids = []

    for input_example in input_examples:
        text, label_strs = input_example["sentence"], input_example["token_label"]
        input_texts.append(text)
        input_labels_str.append(label_strs)
        offset_mappings.append(input_example["offset_mapping"])
        char_labels.append(input_example["char_label"])
        pub_ids.append(input_example["pub_id"])

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
    return input_ids, token_type_ids, attention_mask, input_labels, offset_mappings, input_texts, char_labels, pub_ids


def test_epoch(dataloader, model):
    total_loss = 0.0

    model.eval()
    all_token_predictions = []
    all_token_labels = []

    cur_sen = 0

    fp = open(init_config.output_fn, 'w')

    tepoch = tqdm(dataloader, unit="batch")
    for batch in tepoch:
        tepoch.set_description(f"Test")
        with torch.no_grad():
            input_ids = batch[0].to(config.device)
            token_type_ids = batch[1].to(config.device)
            attention_mask = batch[2].to(config.device)
            labels = batch[3].to(config.device)
            offset_mappings = batch[4]
            input_texts = batch[5]
            char_labels = batch[6]
            pub_ids = batch[7]

            inputs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
                "labels": labels,
            }

            outputs = model(**inputs)

            loss, logits = outputs[:2]
            total_loss += loss.item()

            token_predictions = logits.argmax(dim=2)  # logits
            token_predictions = token_predictions.detach().cpu().numpy()

            for token_prediction, label, offset_mapping, input_text, char_label, pub_id in zip(token_predictions,
                                                                                               labels, offset_mappings,
                                                                                               input_texts, char_labels,
                                                                                               pub_ids):

                cur_sen += 1

                fp.write(pub_id + "\t" + input_text + "\n")

                index_seg = 1
                last_index = len(offset_mapping) - 2

                tmp_label_char = list(' ' * len(input_text))
                for i, cur_label in enumerate(char_label):
                    if cur_label == 'O':
                        tmp_label_char[i] = ' '
                    elif cur_label == 'B-Disease':
                        tmp_label_char[i] = 'D'
                    elif cur_label == 'I-Disease':
                        tmp_label_char[i] = 'd'
                    elif cur_label == 'B-Formula':
                        tmp_label_char[i] = 'P'
                    elif cur_label == 'I-Formula':
                        tmp_label_char[i] = 'p'
                    else:
                        sys.exit("for i, cur_label in enumerate(char_label):")

                tmp_pred_char = list(' ' * len(input_text))
                char_pred = token_pred_to_char_pred(token_prediction, label, offset_mapping)

                for i, cur_label in enumerate(char_pred):
                    if cur_label == 'O':
                        tmp_pred_char[i] = ' '
                    elif cur_label == 'B-Disease':
                        tmp_pred_char[i] = 'D'
                    elif cur_label == 'I-Disease':
                        tmp_pred_char[i] = 'd'
                    elif cur_label == 'B-Formula':
                        tmp_pred_char[i] = 'P'
                    elif cur_label == 'I-Formula':
                        tmp_pred_char[i] = 'p'
                    else:
                        sys.exit("for i, cur_label in enumerate(char_pred):")

                while index_seg <= last_index:
                    if token_prediction[index_seg] == 0 or token_prediction[index_seg] == 1:
                        if token_prediction[index_seg] == 0:
                            cur_entity_type = 'disease'
                        elif token_prediction[index_seg] == 1:
                            cur_entity_type = 'disease(in)'

                        cur_start = offset_mapping[index_seg][0]
                        index_tmp = index_seg
                        while True:
                            if (index_tmp + 1) > last_index or token_prediction[index_tmp + 1] != 1:
                                break
                            index_tmp += 1
                        cur_end = offset_mapping[index_tmp][1]
                        tmp_str = input_text[cur_start:cur_end]

                        fp.write(cur_entity_type + "\t" + tmp_str + "\t" + str(cur_start) + "\t" + str(cur_end) + "\n")
                        index_seg = index_tmp
                    elif token_prediction[index_seg] == 2 or token_prediction[index_seg] == 3:
                        if token_prediction[index_seg] == 2:
                            cur_entity_type = 'prescription'
                        elif token_prediction[index_seg] == 3:
                            cur_entity_type = 'prescription(in)'

                        cur_start = offset_mapping[index_seg][0]
                        index_tmp = index_seg                       
                        while True:
                            if (index_tmp + 1) > last_index or token_prediction[index_tmp + 1] != 3:
                                break
                            index_tmp += 1
                        cur_end = offset_mapping[index_tmp][1]
                        tmp_str = input_text[cur_start:cur_end]

                        fp.write(cur_entity_type + "\t" + tmp_str + "\t" + str(cur_start) + "\t" + str(cur_end) + "\n")
                        index_seg = index_tmp
                    elif token_prediction[index_seg] != 4:
                        sys.exit("Improper token class - I(Inner) tag")

                    index_seg += 1
                fp.write("\n")
                char_prediction = token_pred_to_char_pred(token_prediction, label, offset_mapping)

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

    fp.close()

    token_result = classification_report(all_token_labels, all_token_predictions, digits=4)
    token_f1 = f1_score(all_token_labels, all_token_predictions, average="macro")
    print(token_result)

    tepoch.set_postfix(loss=total_loss / len(dataloader), token_f1=token_f1)

    return total_loss / len(dataloader), token_f1

init_config = define_argparser()

checkpoint = torch.load(init_config.path_model)
train_config = checkpoint['config']
init_config.model_name = train_config.model_name

labels = ["B-Disease", "I-Disease", "B-Formula", "I-Formula", "O"]
label2id = {label: i for i, label in enumerate(labels)}
id2label = {i: label for label, i in label2id.items()}

max_length = init_config.max_seq_len
batch_size = init_config.batch_size

torch.manual_seed(init_config.seed)
np.random.seed(init_config.seed)

config = BertConfig.from_pretrained(init_config.model_name, num_labels=len(label2id))
config.update(init_config.__dict__)


if 'electra' in config.model_name:
    model_loader = ElectraForTokenClassification
elif 'deberta' in config.model_name:
    model_loader = DebertaForTokenClassification
    # model_loader = DebertaV2ForTokenClassification
else:
    model_loader = BertForTokenClassification

model = model_loader.from_pretrained(config.model_name, config=config)
model.cuda()

tokenizer = AutoTokenizer.from_pretrained(init_config.model_name)

dataset_test = load_data(init_config.test_data, tokenizer)

test_dataset = NerDataset(
    tokenizer=tokenizer,
    examples=dataset_test,
    max_length=max_length
)

test_dataloader = DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=collate_fn
)

model.to(init_config.device)
model.load_state_dict(checkpoint['model'])

test_loss, token_f1 = test_epoch(test_dataloader, model)
