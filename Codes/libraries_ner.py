import numpy as np
import torch

from pathlib import Path
from pprint import pprint
from transformers import AutoModel, AutoTokenizer
from transformers import PreTrainedTokenizer, BertConfig, BertForTokenClassification
from typing import Dict, List, Union
from tqdm import tqdm, trange
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report
from torch.utils.data import DataLoader, Dataset


def load_data(file_path: str, tokenizer: PreTrainedTokenizer = None, max_length: int = 128):
    '''read and tokenize dataset from the file'''
    # Read whole text from file
    tmp_data = Path(file_path)
    tmp_text = tmp_data.read_text().strip()
    
    # Separate text into records (sentence text)
    documents = tmp_text.split("\n\n")
    data_list = []
    for doc in documents:
        tmp_lines = doc.split("\n")
        tmp_id, sentence = tmp_lines[0].split("\t")
        char_labels = ["O"] * len(sentence)

        if len(tmp_lines) > 1:
            for i in range(2, len(tmp_lines)):
                tmp_type,_,str_start,str_end = tmp_lines[i].split("\t")
                tmp_start = int(str_start)
                tmp_end = int(str_end)
                
                if tmp_type == "disease":
                    for tmp_i in range(tmp_start, tmp_end):
                        char_labels[tmp_i] = "I-Disease"
                    char_labels[tmp_start] = "B-Disease"
                elif tmp_type == "formula":
                    for tmp_i in range(tmp_start, tmp_end):
                        char_labels[tmp_i] = "I-Formula"
                    char_labels[tmp_start] = "B-Formula"

        # Tokenize string and calculate offsets.
        offset_mappings = tokenizer(sentence,
                                    max_length=max_length,
                                    return_offsets_mapping=True,
                                    truncation=True)["offset_mapping"]
        
        # Assign label of the token
        token_labels = []
        for offset in offset_mappings:
            start, end = offset
            if start == end == 0:
                continue
                
            # Assign the first character label of a token as the label of the token.
            token_labels.append(char_labels[start])

        instance = {
            "pub_id": tmp_id,
            "sentence": sentence,
            "token_label": token_labels,
            "offset_mapping": offset_mappings,
            "char_label": char_labels
        }
        data_list.append(instance)

    return data_list


class NerDataset(Dataset):
    def __init__(
            self,
            tokenizer: PreTrainedTokenizer,
            examples: List,
            max_length: int,
            shuffle: bool = False,
            **kwargs
    ):
        self.dataset = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        instance = self.dataset[index]

        return instance

