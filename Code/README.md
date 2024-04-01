# Application codes for TFDR corpus

This repository offers fine-tuning codes tailored for BERT-based models, optimized for various biomedical text mining tasks like biomedical named entity recognition, key-sentence recognition and relation extraction.

## Named Entity Recognition (NER)

To be added


## Key-Sentence Recognition (KSR) and Relation Extraction (RE)

Please download and extract the pre-processed KSR or RE datasets available in the corpus directory. Then, execute the following command to run the fine-tuning code with default arguments.

```
python run_ner_train.py \
    --model_fn \
    --train_fn \
    --pretrained_model_name=dmis-lab/biobert-base-cased-v1.2 \
    --valid_ratio=.2 \
    --batch_size_per_device=16 \
    --n_epochs=8 \
    --warmup_ratio=.5 \
    --max_length=128
```

The execution of prediction and evaluation code proceeds as follows with default arguments:

```
python run_ner_test.py \
    --model_fn \
    --eval_fn \
    --output_fn \
    --gpu_id=-1 \
    --batch_size=32 \
    --top_k=1 \
    --max_length=128
```
