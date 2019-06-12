# RankQA: The Re-Ranking Module

This part of the repository contains code to train and evaluate the re-ranking module based on pre-computed and aggregated features of candidate-answers.

## Quick Setup

1. Install requirements in `requirements.txt`. The file also provides a complete list of python packages that were installed in the conda environment used during implementation.
2. Download features from [HERE](https://mgtvgsgb.livedrive.com/item/0987ec25f8c044bfa0ba738e7e025f1c) (compressed 3.6 GB). You can find more details about those features below.
3. Extract the compressed file to `./features`
4. Create an empty folder `out`
5. Run `python run.py` 

Alternatively, you find a pre-trained model in `./pre-trained-model`

## Pre-computed Answer Candidates

Feature generation is computationally intense since you need to run the prediction pipeline for all 200.000+ question-answer pairs on the full Wikipedia. 
Therefore we provide precomputed and aggregated answer candidates for the following datasets:

* SQuAD (train/dev)
* WebQuestions (train/test)
* WikiMovies (train/test)
* CuratedTrec (train/test)

You can download the features from [HERE](https://mgtvgsgb.livedrive.com/item/0987ec25f8c044bfa0ba738e7e025f1c) (compressed 3.6 GB).

*NOTE*: Although we only use the features described in the paper we provide the tokenized question, paragraph, and answer along with every candidate answer in order to allow future research with more complex networks. 

### Data Format

Pre-computed features were generated in batches of up to 1000 questions. E.g., you find features for WebQuestions-train in three separate files `WebQuestions-test-default-pipeline.preds-part-0` to `WebQuestions-test-default-pipeline.preds-part-2`.
Every line in a file contains the top-k aggregated candidate answers for one question: 

```python
[{cand-1}, {cand-2}, ... ,{cand-k}] # candidates for question 1
...
[{cand-1}, {cand-2}, ... ,{cand-k}] # candidates for question 1000
```

Every candidate answer is JSON encoded and contains the the features described in the paper as well as tokenized context, question, and answer:

```
KEY                 VALUE
------------------------------------------------------------------------------
span_score          score for the answer as determined by the MC module
sum_span_score      aggregated features for span score
min_span_score      
max_span_score
avg_span_score
doc_score           score of the document the answer was extracted from
min_doc_score       aggregated document scores
max_doc_score
avg_doc_score
sum_doc_score
first_occ           first position the answer occurred in the original ranking
num_occ             how often did the answer occur before aggregation
context_len         length of the context (the paragraph that contains the answer)
question_len        length of the question 
qid                 id of question unique in dataset and split
doc_id              id of the document that contains the answer
span                the textual answer
span_tokens         tokenized answer
span_ner            named entities in answer
span_pos            POS tags in answer
question            the question
question_tokens     tokenized question
context
 | -> text          the paragraph that contains the answer
 | -> start         start of answer within the paragraph
 | -> end           end of answer within the paragraph
 | -> tokens        tokenized paragraph
```
