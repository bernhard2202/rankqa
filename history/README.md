# Changelog and History

- We addressed the reviewers' comments and submitted the camera ready version on June 3: [ACL Changelog](#acl-2019-camera-ready-changelog)
- The paper was accepted to ACL 2019 on May 13 2019: [Meta-Review](#acl-meta-review)
- The paper was submtited to ACL 2019 on March 4 2019: [Reviews](#acl-reviews)

## ACL 2019 Camera Ready Changelog

How we addressed the reviews:

- We added the results of a perfect re-ranking system (upper bound)
- We implemented a second QA pipeline based on BERT to show that results are robust across implementations
- We broke down the feature importance analysis from three groups of features to seven smaller logical groups
- We added analysis whether RankQA is capable of keeping the set of correctly answered questions after re-ranking 
- Formulations that were found misleading and typos. 
- We included the results of Min et al. (2018)

Time did not suffice but we will still try to do the following: 

- Comparison with [Wang et al. 2017](https://arxiv.org/pdf/1711.05116.pdf). This system uses re-ranking based on sentences that are fetched ahead of time and does not scale to paragraph-level retrieval on the entire Wikipedia. We are still trying to find a way to make a fir comparison. 
- Deeper error analysis. 

## ACL Meta Review
The paper presents a neural model to solve answer re-ranking task in QA. The main contribution is to explore several features, including IR features, MC features and linguistic features, to improve selection performance. The experimental results on several open accessed QA dataset (MC, RetrievalQA, KBQA) show that the proposed approach could obtain positive results.

Based on all the reviewers' comments, the whole paper is clear and easy to follow. The proposed method is elegant and straightforward although it is simple. The experimental parts are sufficient. Most reviewers give it positive socres. So, I think that this paper could be accepted.


## ACL reviews

### Reviewer 1

**_What is this paper about, what contributions does it make, what are the main strengths and weaknesses?_**


This paper proposes the inclusion of a new stage in neural QA systems. This new step is based on ranking candidate answers, which was a typical step in old QA systems. The authors compare their results with those from other systems with and without adding the new step, obtaining better results with the new step. 

Although the proposal and results are interesting, I miss a deeper analysis of results. The authors should: 

1) analyze the main features of questions where their system helps with respect to other proposals.

2) analyze questions wrongly answered by their proposal

3) study the impact of answer aggregation

4) does your system answer correctly all the questions correctly answered by DrQA (the baseline of this paper)? I mean, it is remarkable to detect if you keep the set of correct answers and detect in what questions your proposal helps

5) include results of a perfect ranking system. These results would be the upper bound for your proposal. Then, you can show how close you are from this upper bound.

I think all these observations are essential in this paper to understand the main contributions and future work.


The main contributions of this paper are:
    - the proposal of the new stage in neural QA and how to include it in current architectures
    - the loss function for the new stage
    - the results comparing the new proposal over several datasets


main STRENGTHS:
- the inclusion of a new stage for ranking answers in neural-QA
- the loss function for the new stage
- the analysis of results using several datasets

main WEAKNESS:
- the analysis of errors must be deeper
- another baseline should be Bert
- the paper from Min et al. (2018) include similar results in Squad-open, but these results are not included in your work.
- I miss future work


**_Reasons to accept_**

The suggestion of the new stage and the proper definition of a loss function for the new stage. Besides, the fact that the new stage contributes to obtaining better results over several datasets.


**_Reasons to reject:_**

I miss a deeper analysis of questions wrongly answered and questions that are correctly answered with the new stage.


**_Reviewer's Scores_**

 Overall Recommendation: 3.5

**_Typos, Grammar, and Style_**

- In page 2: "RankQA presents the first QA pipeline..." -> "RankQA presents the first neural QA pipeline...". There have been several proposals with answer ranking. So, I think you mean neural QA
- page 7, first line: "coprus" -> "corpus"

### Reviewer 2
**_What is this paper about, what contributions does it make, what are the main strengths and weaknesses?_**

The authors propose a module that does answer re-ranking, the module takes features from information retrieval (IR) and machine comprehension (MC) modules as input, a neural network ranks all answers come from MC module. 
In this work, the authors explore a way of leveraging IR features and some linguistic features (e.g., part-of-speech and named entity) to improve answer selection performance.

The paper is well written, the code in supplements is clear. 
The proposed answer re-ranking module is simple yet effective, the authors show with answer re-ranking module as add-on, DrQA can outperform current state-of-the-art models on 3 out of 4 datasets by a non-trivial margin. 

I like the idea described in this paper, but there are still several points I'm not clear, which I list below:
1. As mentioned in Eqn. (1), the re-ranking network is quite simple, I'm wondering is it because the authors have tried other structures and this turned out to be the one worked best? 
2. The ablation experiments as described in Table 3 is a bit too high level, it would be interesting to see what specific features are the most helpful for answer re-ranking.
3. I would be curious to see some examples how the answer rankings change before and after applying the answer re-ranking module.

Overall I like this work and I think the quality is above borderline.


**_Reasons to accept_**

Well written paper with implementation provided. Method is simple but seems elegant (a simple add-on neural module improves performance by quite a bit).


**_Reviewer's Scores_**

Overall Recommendation: 3.5

_**Typos, Grammar, and Style**_

Line 747 --- "and and"

### Reviewer 3 

**_What is this paper about, what contributions does it make, what are the main strengths and weaknesses?_**

The authors propose a reranking module in a retrieval-based QA pipeline, which takes account into features from both the retrieval features, MC features and aggregation features. The proposed model achieves significant improvements.

Strengths:
1. The paper is executed well. The improvements are descent and ablation studies indeed show that both retrieval-based features, MC features and aggregation features are useful. Hence the motivation of a better fusion between retrieval and MC makes sense.
2. The paper is well-written and easy to understand.

Weaknesses:
1. As mentioned by the authors, the reranking idea is also proposed in [Wang et al. 2017](https://arxiv.org/pdf/1711.05116.pdf) The difference is that they use neural networks for reranking while the authors use hand-engineered features for reranking. The authors mention that the weakness of their model is that you need to use a neural network to extract features which is time consuming, but since the MC module is already a neural network, would adding a reranking neural network change the time consumption a lot? It would be great if the authors can compare to their method.
2. I am not fully convinced that the benefits of reranking is fusing information-retrieval features. In theory, your MC module also can read those features and take it into account. Is the method better because it's easy to optimize?

**_Reasons to accept_**

The paper is well executed and the idea is interesting.


**_Reasons to reject:_**

The paper could have done a better job in studying and explaining where the improvements come from. The baselines can be improved by replicating Wang et al. 2017.



**_Reviewer's Scores_**

Overall Recommendation: 4
