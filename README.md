# UMR: Unsupervised Multilingual Dense Retrieval via Generative Pseudo Labeling

This repository contains the source code of our paper "Unsupervised Multilingual Dense Retrieval via Generative Pseudo Labeling", which has been accepted to Findings of EACL 2024.

<img width="1024" alt="image" src="https://github.com/MiuLab/UMR/assets/11765276/8a29f951-eeea-4927-aa73-44308d625403">

<img width="1024" alt="image" src="https://github.com/MiuLab/UMR/assets/11765276/0b72b944-79e4-4dd5-bf98-289d003a6970">



## Requirements
* Python >= 3.8
* transformers
* torch

Please install all required packages listed in `requirements.txt` by running the following command:
```bash
pip install -r requirements.txt
```

## Data
We use the [XOR-TYDI QA](https://github.com/AkariAsai/XORQA) dataset in our experiments, which includes **XOR-Retrieve** and **XOR-Full**. Please download the datasets from the following link and put it in the `data` directory:

For **XOR-Retrieve**, where a question is written in the target language (e.g., Japanese) and a system is required to retrieve English document that answers the question:
- [Train data](https://nlp.cs.washington.edu/xorqa/XORQA_site/data/xor_train_retrieve_eng_span.jsonl)
- [Development data](https://nlp.cs.washington.edu/xorqa/XORQA_site/data/xor_dev_retrieve_eng_span_v1_1.jsonl)
- [Test data (Question Only)](https://nlp.cs.washington.edu/xorqa/XORQA_site/data/xor_test_retrieve_eng_span_q_only_v1_1.jsonl)
- [Collection (13G)](https://nlp.cs.washington.edu/xorqa/XORQA_site/data/models/enwiki_20190201_w100.tsv)


For **XOR-Full**, where a question is written in the target language (e.g., Japanese) and a system is required to retrieve from multilingual documents and output a short answer in the target language:
- [Train data](https://nlp.cs.washington.edu/xorqa/XORQA_site/data/xor_train_full.jsonl)
- [Development data](https://nlp.cs.washington.edu/xorqa/XORQA_site/data/xor_dev_full_v1_1.jsonl)
- [Test data (Question Only)](https://nlp.cs.washington.edu/xorqa/XORQA_site/data/xor_test_full_q_only_v1_1.jsonl)
- [Collection (29GB)](https://nlp.cs.washington.edu/xorqa/cora/models/all_w100.tsv)


## Trained Checkpoints and Processed Data
We provide the trained checkpoints and processed data for the UMR model. You can download the files from the [Google Drive](https://drive.google.com/drive/folders/1imebGaCvLky9mujcTggn0wmQIVXHwPNs?usp=sharing).


## Training
Training the UMR model consists of two steps: unsupervised multilingual reranking and knowledge-distilled retriever training. We provide the commands for training the UMR model below. We trained one retriever for each task. The following commands are examples of training the UMR model using the XOR-Retrieve dataset. You can modify the commands to train the UMR model using the XOR-Full dataset.

### Step 1: Unsupervised Multilingual Reranking

#### Retrieve Top-k Passages with an Initial Retriever (mContriever)
First, we generate context embeddings using the mContriever model.
```bash
python3 generate_dense_embeddings.py \
  --pretrained_model_cfg facebook/mcontriever \
  --encoder_model_type hf_bert \
  --sequence_length 256 \
  --batch_size 256 \
  --ctx_file data/enwiki_20190201_w100.tsv \
  --shard_id 0 --num_shards 1 \
  --out_file data/enwiki_embeddings_iter0 \
  --fp16
```
If using a trained checkpoint, e.g., in the second iteration, specify the `--model_file` argument to the checkpoint file.

Then, we retrieve top-k passages for each question using the mContriever model.
```bash
python3 dense_retriever.py \
  --pretrained_model_cfg facebook/mcontriever \
  --encoder_model_type hf_bert \
  --sequence_length 256 \
  --ctx_file data/enwiki_20190201_w100.tsv \
  --qa_file data/xor_train_retrieve_eng_span.jsonl \
  --encoded_ctx_file "data/enwiki_embeddings_iter0*" \
  --out_file data/xor_retrieve_train_retrieved_iter0.json \
  --n-docs 100 \
  --validation_workers 1 --batch_size 128 --search_batch_size 512
```
If using a trained checkpoint, e.g., in the second iteration, specify the `--model_file` argument to the checkpoint file.

#### Rerank Top-k Passages with an LM (mt5-xl)
```bash
python3 -m torch.distributed.launch --nproc_per_node {NGPUS} upr-multi.py \
  --num-workers 2 \
  --shard-size 2 \
  --topk-passages 100 \
  --hf-model-name "chaoweihuang/mt5-xl-lm-adapt" \
  --use-gpu \
  --use-fp16 \
  --report-topk-accuracies 1 5 20 100 \
  --retriever-topk-passages-path data/xor_retrieve_train_retrieved_iter0.json \
  --reranker-output-dir data/xor_retrieve_train_retrieved_iter0_reranked
```
The reranked results will be saved in the `data/xor_retrieve_train_retrieved_iter0_reranked/rank{RANK}.json`. You will need to merge the results from different ranks to obtain the final reranked results.


### Step 2: Knowledge-Distilled Retriever Training
Once the reranked results are obtained, we can train the knowledge-distilled retriever using the reranked results. The following command is an example of training the knowledge-distilled retriever using the XOR-Retrieve dataset. You may want to split the reranked results into training and development sets and modify the `--train_file` and `--dev_file` arguments accordingly.
```bash
CUDA_VISIBLE_DEVICES=${DEVICES} python3 train_dense_encoder_with_llm.py \
    --max_grad_norm 2.0 \
    --encoder_model_type hf_bert \
    --pretrained_model_cfg facebook/mcontriever \
    --seed 12345 \
    --sequence_length 256 \
    --warmup_steps 1237 \
    --num_contexts 16 \
    --batch_size 16 \
    --gradient_accumulation_steps 1 \
    --inbatch_negative \
    --temperature 10 \
    --train_file data/xor_retrieve_train_retrieved_iter0_reranked/rank0.json \
    --dev_file {DEV_FILE} \
    --output_dir {CHECKPOINT_DIR} \
    --learning_rate 2e-05 \
    --num_train_epochs 10 \
    --dev_batch_size 12 \
    --val_av_rank_start_epoch 0 \
    --global_loss_buf_sz 2000000 \
    --eval_per_epoch 4 \
    --grad_cache \
    --q_chunk_size 16 \
    --ctx_chunk_size 8 \
    --restart \
    --fp16 \
    --wandb_project {WANDB_PROJECT} \
    --wandb_name {WANDB_NAME}
```

This shows one iteration of the UMR training. To train the UMR model for more iterations, you need to repeat the above steps using the trained checkpoint.


## Evaluation
We provide the commands for evaluating the UMR model on the XOR-Retrieve dataset. You can modify the commands to evaluate the UMR model on the XOR-Full dataset. Note that we provide the commands for evaluating on the development set as the test set is not publicly available.

### Step 1: Generate Embeddings for the Dev Set
```bash
python generate_dense_embeddings.py \
    --model_file {CHECKPOINT_FILE} \
    --encoder_model_type hf_bert \
    --sequence_length 256 \
    --batch_size 256 \
    --ctx_file data/enwiki_20190201_w100.tsv \
    --shard_id 0 --num_shards 1 \
    --out_file data/enwiki_embeddings_iter1 \
    --fp16
```

### Step 2: Retrieve Top-k Passages for the Dev Set
```bash
python3 dense_retriever.py \
    --model_file {CHECKPOINT_FILE} \
    --encoder_model_type hf_bert \
    --sequence_length 256 \
    --ctx_file data/enwiki_20190201_w100.tsv \
    --qa_file data/xor_dev_retrieve_eng_span_v1_1.jsonl \
    --encoded_ctx_file "data/enwiki_embeddings_iter1*" \
    --out_file data/xor_retrieve_dev_retrieved_iter1.json \
    --n-docs 100 \
    --validation_workers 1 --batch_size 128 --search_batch_size 1024
```

### Step 3: Evaluate the Results
```bash
python3 evals/eval_xor_retrieve.py \
  --pred_file data/xor_retrieve_dev_retrieved_iter1.json \
  --data_file data/xor_dev_retrieve_eng_span_v1_1.jsonl
```
Note that for evaluating on the XOR-Full dataset, since there is no ground truth for the retrieval task, we feed the retrieval results from UMR to the CORA reader (mGEN) and evaluate the end-to-end QA performance. Please refer to the [CORA](https://github.com/AkariAsai/CORA) repository for how to run the mGEN model and evaluate the QA performance.


## Reference
If you find our work useful, please cite the following paper:
```
    @inproceedings{huang2024umr,
        title = "Unsupervised Multilingual Dense Retrieval via Generative Pseudo Labeling",
        author = "Huang, Chao-Wei and Hsu, Tsu-Yuan and Li, Chen-An and Hsu, Chen-Yu and Chen, Yun-Nung",
        booktitle = "Proceedings of the 18th Conference of the European Chapter of the Association for Computational Linguistics",
        month = mar,
        year = "2024",
        publisher = "Association for Computational Linguistics",
    }
```
