##
## This codes are modified for this study 
## from the base codes in << https://github.com/kh-kim/simple-ntc >>
##

import argparse
import random
import torch
import pandas as pd
import os

from datasets import Dataset

from transformers import BertForSequenceClassification
from transformers import AutoTokenizer
from transformers import TrainingArguments
from transformers import Trainer
from transformers import DataCollatorWithPadding

from sklearn.metrics import accuracy_score


def define_argparser():
    '''process the arguments in the commandline'''
    p = argparse.ArgumentParser()    
    p.add_argument('--model_fn', required=True)
    p.add_argument('--train_fn', required=True)
    p.add_argument('--pretrained_model_name', type=str, default='allenai/scibert_scivocab_uncased')
    p.add_argument('--valid_ratio', type=float, default=.1)
    p.add_argument('--batch_size_per_device', type=int, default=8)
    p.add_argument('--n_epochs', type=int, default=8)
    p.add_argument('--warmup_ratio', type=float, default=.2)
    p.add_argument('--max_length', type=int, default=128)
    config = p.parse_args()

    return config


def get_datasets(fn, valid_ratio=.2):
    '''read, shuffle, and split training dataset from the file'''
    ## dataset should have no header and tab delimiter
    ## first column: labels
    ## second clumn: texts
    
    df = pd.read_csv(fn, sep = '\t', keep_default_na = False, header = None, dtype={0: str})
    df.rename(columns = {0:'labels', 1:'text'}, inplace = True) 
    unique_labels = sorted(set(df['labels']))
    label_to_index = {}
    index_to_label = {}
    for i, label in enumerate(unique_labels):
        label_to_index[label] = i
        index_to_label[i] = label
    df['labels'] = list(map(label_to_index.get, df['labels']))

    dataset = Dataset.from_dict(df)
    shuffled_splitted_dataset = dataset.train_test_split(test_size = valid_ratio, seed = 27)
    train_dataset = shuffled_splitted_dataset['train']
    valid_dataset = shuffled_splitted_dataset['test']
    
    return train_dataset, valid_dataset, label_to_index, index_to_label


def main(config):
    train_dataset, valid_dataset, label_to_index, index_to_label = get_datasets(config.train_fn, config.valid_ratio)
    print('|train| =', len(train_dataset), '\n|valid| =', len(valid_dataset))

    total_batch_size = config.batch_size_per_device * torch.cuda.device_count()
    n_total_iterations = int(len(train_dataset) / total_batch_size * config.n_epochs)
    n_warmup_steps = int(n_total_iterations * config.warmup_ratio)
    print('#total_iterations =', n_total_iterations, '\n#warmup_steps =', n_warmup_steps)

    model_loader = BertForSequenceClassification
    model = model_loader.from_pretrained(config.pretrained_model_name, num_labels=len(index_to_label), label2id=label_to_index, id2label=index_to_label)
    tokenizer = AutoTokenizer.from_pretrained(config.pretrained_model_name)
    
    def preprocess_tokenizer(batch):
        return tokenizer(batch["text"], padding = True, truncation = True, max_length = config.max_length, return_tensors = "pt")
    
    tokenized_train_dataset = train_dataset.map(preprocess_tokenizer, batched=True, remove_columns = ["text"])
    tokenized_valid_dataset = valid_dataset.map(preprocess_tokenizer, batched=True, remove_columns = ["text"])

    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        return {'accuracy': accuracy_score(labels, preds)}
    
    training_args = TrainingArguments(
        output_dir='./.checkpoints',
        num_train_epochs = config.n_epochs,
        per_device_train_batch_size = config.batch_size_per_device,
        per_device_eval_batch_size = config.batch_size_per_device,
        warmup_steps = n_warmup_steps,
        weight_decay = 0.01,
        fp16 = True,
        evaluation_strategy = 'epoch',
        save_strategy = 'epoch',
        logging_steps = n_total_iterations // 10,
        save_steps = n_total_iterations // config.n_epochs,
        load_best_model_at_end = True,
    )
    
    trainer = Trainer(
        model = model,
        args = training_args,
        data_collator = DataCollatorWithPadding(tokenizer),
        train_dataset = tokenized_train_dataset,
        eval_dataset = tokenized_valid_dataset,
        compute_metrics = compute_metrics,
        tokenizer = tokenizer
    )

    trainer.train()

    torch.save({
        'config': config,
        'model': trainer.model.state_dict(),
        'class': index_to_label
    }, config.model_fn)


if __name__ == '__main__':
    config = define_argparser()
    main(config)