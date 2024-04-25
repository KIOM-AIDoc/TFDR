##
## This codes are modified for this study 
## from the base codes in << https://github.com/kh-kim/simple-ntc >>
##

import argparse
import torch
import csv
import numpy as np

import torch.nn.functional as F
from transformers import AutoTokenizer
from transformers import BertForSequenceClassification
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report


def define_argparser():
    '''process the arguments in the commandline'''
    p = argparse.ArgumentParser()
    p.add_argument('--model_fn', required=True)
    p.add_argument('--eval_fn', required=True)
    p.add_argument('--output_fn', required=True)
    p.add_argument('--gpu_id', type=int, default=-1)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--top_k', type=int, default=1)
    p.add_argument('--max_length', type=int, default=128)    
    config = p.parse_args()

    return config


def read_text(fn):
    '''read test dataset from the file for the evaluation'''
    ## dataset should have no header and tab delimiter
    ## first column: labels
    ## second clumn: texts
    
    labels, texts = [], []
    with open(fn, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line.strip() != '':
                label, text = line.strip().split('\t')
                labels += [label]
                texts += [text]
    return labels, texts


def main(config):
    saved_data = torch.load(config.model_fn, map_location='cpu' if config.gpu_id < 0 else 'cuda:%d' % config.gpu_id)
    train_config = saved_data['config']
    bert_best = saved_data['model']
    index_to_label = saved_data['class']

    true_labels, texts = read_text(config.eval_fn)

    with torch.no_grad():
        tokenizer = AutoTokenizer.from_pretrained(train_config.pretrained_model_name)
        model_loader = BertForSequenceClassification        
        model = model_loader.from_pretrained(train_config.pretrained_model_name, num_labels=len(index_to_label))
        model.load_state_dict(bert_best)

        if config.gpu_id >= 0:
            model.cuda(config.gpu_id)
        device = next(model.parameters()).device

        model.eval()

        y_hats = []
        for idx in range(0, len(texts), config.batch_size):
            mini_batch = tokenizer(
                texts[idx:idx + config.batch_size],
                padding=True,
                truncation=True,
                max_length=config.max_length,
                return_tensors="pt"
            )
            x = mini_batch['input_ids']
            x = x.to(device)
            mask = mini_batch['attention_mask']
            mask = mask.to(device)

            y_hat = F.softmax(model(x, attention_mask = mask).logits, dim = -1)
            y_hats += [y_hat]
            
        # concatenate the mini-batch wise result
        y_hats = torch.cat(y_hats, dim=0)
        probs, indice = y_hats.cpu().topk(config.top_k)
        
        pred_labels = []
        for i in range(len(texts)):
            pred_labels.extend([index_to_label[int(indice[i][j])] for j in range(config.top_k)])

        # write inference results into the file
        with open(config.output_fn, 'w', encoding='utf-8', newline='') as f:
            tw = csv.writer(f, delimiter='\t')
            for i in range(len(texts)):
                tw.writerow([pred_labels[i], texts[i]])

        print(classification_report(true_labels, pred_labels, digits=4, zero_division=0.0))                

        
if __name__ == '__main__':
    config = define_argparser()
    main(config)
