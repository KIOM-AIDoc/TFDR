# Application codes for TFDR corpus

This repository offers fine-tuning codes tailored for BERT-based models, optimized for various biomedical text mining tasks like biomedical Named Entity Recognition (NER), Key-Sentence Recognition (KSR) and Relation Extraction (RE).

## NER codes

Please download and extract the pre-processed NER datasets available in the corpus directory. Then, execute the following command to run the fine-tuning code with default arguments.

```
python run_ner_train.py \
    --model_fn \
    --train_fn \
    --pretrained_model_name=dmis-lab/biobert-v1.1 \
    --valid_ratio=.2 \
    --batch_size_per_device=16 \
    --n_epochs=30 \
    --max_seq_len=400 \
    --learning_rate=.5 \
    --adam_epsilon=128 \
    --device=cuda \
    --max_grad_norm=1.0 \
    --seed=1234
```

The execution of prediction and evaluation code proceeds as follows with default arguments:

```
python run_ner_test.py \
    --model_fn \
    --eval_fn \
    --output_fn \
    --device=cuda \
    --batch_size=16 \
    --max_seq_len=400 \
    --seed=1234
```


## KSR and RE codes

Please download and extract the pre-processed KSR or RE datasets available in the corpus directory. Then, execute the following command to run the fine-tuning code with default arguments.

```
python run_ksr_re_train.py \
    --model_fn \
    --train_fn \
    --pretrained_model_name=allenai/scibert_scivocab_uncased \
    --valid_ratio=.2 \
    --batch_size_per_device=8 \
    --n_epochs=8 \
    --warmup_ratio=.5 \
    --max_length=128
```

The execution of prediction and evaluation code proceeds as follows with default arguments:

```
python run_ksr_re_test.py \
    --model_fn \
    --eval_fn \
    --output_fn \
    --gpu_id=-1 \
    --batch_size=8 \
    --top_k=1 \
    --max_length=128
```
