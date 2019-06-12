# BERT-QA

We designed a customized QA pipeline based on [DrQA](https://github.com/facebookresearch/DrQA/) and [BERT](https://github.com/google-research/bert) to show robustness across implementations. 

**IMPORTANT:** The code here is heavily based on the original GitHub repositories listed below. It was implemented as a proof-of-concept and add-on to the main repository, rather than a fully optimized QA pipeline. Therefore, some things are not optimally parametrized, commented, etc. 

* https://github.com/facebookresearch/DrQA
* https://github.com/google-research/bert

## BERT-QA Pipeline

Our pipeline performs the following steps in order to answer a question: 
1. The information retrieval module of DrQA is used to retrieve the top-n documents from Wikipedia. 
2. The documents are then split into paragraphs.
3. All paragraphs are paired with the question and fed into BERT (if the paragraph is too long we split it with an according stride).
4. Top-k answer candidates are returned, ranked by the unnormalized BERT score (softmax is removed). The code aggregates answers and extracts all features necessary for re-ranking.

## Setup

This code was developed under Python 3.6. 

Python requirements are listed in ``requirements.txt``

### Setup Data and Models

1. Download the BERT-base-cased model and set up the environment variable ``$BERT_BASE_DIR`` as described [HERE](https://github.com/google-research/bert).
2. Pre-train BERT on SQuAD as described in the original repository, store the final model in a folder ``$BERT_FINE_TUNED``.
3. Use the ``download.sh`` script from [HERE](https://github.com/facebookresearch/DrQA/) to download DrQA data. 
4. You can delete the ``reader`` folder from the downloaded data, it is not needed. 
5. Create a folder ``output``.

## Run the Pipeline

You can run the pipeline in order to generate candidate answers for a given dataset using the following command:

```bash
python generate_candidates.py  --vocab_file=${BERT_BASE_DIR}/vocab.txt \
                   --bert_config_file=${BERT_BASE_DIR}/bert_config.json \
                   --output_dir=/tmp \
                   --init_checkpoint=${BERT_FINE_TUNED} \
                   --do_predict=True \
                   --predict_file=./data/datasets/SQuAD-v1.1-train.txt \
                   --retriever_model=./data/wikipedia/docs-tfidf-ngram\=2-hash\=16777216-tokenizer\=simple.npz  \
                   --doc_db=./data/wikipedia/docs.db \
                   --out_name=out/squad_train 
```

Parameters explained:

```
--vodab_file         BERT vocabulary file, comes with the pre-trained model 
--bert_config_file   BERT config file, comes with the pre-trained model
--init-checkpoint    BERT model fine-tuned on squad
--output_dir         a temporary folder for temporary tensorflow outputs
--do_predict         must be True
--predict_file       dataset file to extract the answer candidates for (see below)
--retriever_model    DrQA retriever model
--doc_db             DrQA doc-db storing Wikipedia documents
--out_name           path and name to store the candidate answers to, will be extended by '-feat-batch-N.txt'
```

The input to the pipeline is a QA dataset in `.txt` format where each line is a JSON encoded QA pair:

```python
'{"question": "q1", "answer": ["a11", ..., "a1i"]}'
...
'{"question": "qN", "answer": ["aN1", ..., "aNi"]}'
```

The python file runs batches of 500 questions through the pipeline and writes one `.txt` file to ``--out-name`` per batch. Every output file contains one line per question, where every line itself contains all aggregated ``top-k`` answer candidates for a question.
You can load the generated features in the answer re-ranking part of this repository after removing the code to load NER and POS features.