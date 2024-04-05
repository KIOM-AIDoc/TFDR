# Application codes for TFDR corpus

This repository offers fine-tuning codes tailored for BERT-based models, optimized for various biomedical text mining tasks like biomedical Named Entity Recognition (NER), Key-Sentence Recognition (KSR) and Relation Extraction (RE).

## NER codes

To be added


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
    --batch_size=32 \
    --top_k=1 \
    --max_length=128
```
