# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Run BERT on SQuAD 1.1 and SQuAD 2.0."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import json
import os
import re
import string
import unicodedata
from multiprocessing import Pool as ProcessPool
from multiprocessing.util import Finalize

import modeling
import tensorflow as tf
import tokenization
from bert_util import validate_flags_or_throw, model_fn_builder, FeatureWriter, \
    input_fn_builder, RawResult, write_predictions, check_is_max_context, InputFeatures
from drqa_util import split_doc
from retriever import DocDB
from retriever import TfidfDocRanker

flags = tf.flags

FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")

## Other parameters
flags.DEFINE_string("train_file", None,
                    "SQuAD json for training. E.g., train-v1.1.json")
flags.DEFINE_string("retriever_model", None,
                    "")
flags.DEFINE_string("doc_db", None,
                    "")
flags.DEFINE_string("out_name", None,
                    "")
flags.DEFINE_string(
    "predict_file", None,
    "SQuAD json for predictions. E.g., dev-v1.1.json or test-v1.1.json")

flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer(
    "max_seq_length", 384,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_integer(
    "doc_stride", 128,
    "When splitting up a long document into chunks, how much stride to "
    "take between chunks.")

flags.DEFINE_integer(
    "max_query_length", 64,
    "The maximum number of tokens for the question. Questions longer than "
    "this will be truncated to this length.")

flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("do_predict", False, "Whether to run eval on the dev set.")

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("predict_batch_size", 8,
                     "Total batch size for predictions.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 3.0,
                   "Total number of training epochs to perform.")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_integer(
    "n_best_size", 20,
    "The total number of n-best predictions to generate in the "
    "nbest_predictions.json output file.")

flags.DEFINE_integer(
    "max_answer_length", 30,
    "The maximum length of an answer that can be generated. This is needed "
    "because the start and end predictions are not conditioned on one another.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")
tf.flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

tf.flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")

flags.DEFINE_bool(
    "verbose_logging", False,
    "If true, all of the warnings related to data processing will be printed. "
    "A number of warnings are expected for a normal SQuAD evaluation.")

flags.DEFINE_bool(
    "answer_regex", False,
    "TODO")  # TODO

flags.DEFINE_bool(
    "version_2_with_negative", False,
    "If true, the SQuAD examples contain some that do not have an answer.")

flags.DEFINE_float(
    "null_score_diff_threshold", 0.0,
    "If null_score - best_non_null is greater than the threshold predict null.")

# ------------------------------------------------------------------------------
# Multiprocessing functions to fetch and tokenize text
# ------------------------------------------------------------------------------

PROCESS_DB = None
PROCESS_TOK = None


def init(db_class, db_opts, tok_class, tok_opts):
    global PROCESS_TOK, PROCESS_DB
    PROCESS_DB = db_class(**db_opts)
    PROCESS_TOK = tok_class(**tok_opts)
    Finalize(PROCESS_DB, PROCESS_DB.close, exitpriority=100)


def convert_to_id(tokens):
    global PROCESS_TOK
    return PROCESS_TOK.convert_tokens_to_ids(tokens)


def fetch_text(doc_id):
    global PROCESS_DB
    return PROCESS_DB.get_doc_text(doc_id)


def tokenize_question(text, max_query_length=FLAGS.max_query_length):
    global PROCESS_TOK
    query_tokens = PROCESS_TOK.tokenize(text)
    if len(query_tokens) > max_query_length:
        query_tokens = query_tokens[0:max_query_length]
    return query_tokens


def tokenize_document(text):
    global PROCESS_TOK

    def is_whitespace(c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False

    doc_tokens = []
    prev_is_whitespace = True
    for c in text:
        if is_whitespace(c):
            prev_is_whitespace = True
        else:
            if prev_is_whitespace:
                doc_tokens.append(c)
            else:
                doc_tokens[-1] += c
            prev_is_whitespace = False

    all_doc_tokens = []
    tok_to_orig_index = []
    for (i, token) in enumerate(doc_tokens):
        sub_tokens = PROCESS_TOK.tokenize(token)
        for sub_token in sub_tokens:
            tok_to_orig_index.append(i)
            all_doc_tokens.append(sub_token)

    return all_doc_tokens, tok_to_orig_index, doc_tokens


def normalize(text):
    """Resolve different type of unicode encodings."""
    return unicodedata.normalize('NFD', text)


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(normalize(s)))))


def get_jaccard_sim(str1, str2):
    a = set(str1)  # .split())
    b = set(str2)  # .split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)

    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

    validate_flags_or_throw(bert_config, FLAGS)

    tf.gfile.MakeDirs(FLAGS.output_dir)

    tpu_cluster_resolver = None
    if FLAGS.use_tpu and FLAGS.tpu_name:
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
            FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
    run_config = tf.contrib.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        master=FLAGS.master,
        model_dir=FLAGS.output_dir,
        save_checkpoints_steps=FLAGS.save_checkpoints_steps,
        tpu_config=tf.contrib.tpu.TPUConfig(
            iterations_per_loop=FLAGS.iterations_per_loop,
            num_shards=FLAGS.num_tpu_cores,
            per_host_input_for_training=is_per_host))

    num_train_steps = None
    num_warmup_steps = None

    model_fn = model_fn_builder(
        bert_config=bert_config,
        init_checkpoint=FLAGS.init_checkpoint,
        learning_rate=FLAGS.learning_rate,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        use_tpu=FLAGS.use_tpu,
        use_one_hot_embeddings=FLAGS.use_tpu)

    # If TPU is not available, this will fall back to normal Estimator on CPU
    # or GPU.
    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=FLAGS.use_tpu,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=FLAGS.train_batch_size,
        predict_batch_size=FLAGS.predict_batch_size)

    #
    # DRQA CODE HERE:
    #
    tf.logging.info('load ranker..')
    ranker = TfidfDocRanker(tfidf_path=FLAGS.retriever_model, strict=False)

    tf.logging.info('load doc db ..')
    db_class = DocDB
    db_options = {'db_path': FLAGS.doc_db}
    tok_class = tokenization.FullTokenizer
    tok_options = {'vocab_file': FLAGS.vocab_file, 'do_lower_case': FLAGS.do_lower_case}
    processes = ProcessPool(
        5,  # TODO move to flags
        initializer=init,
        initargs=(db_class, db_options, tok_class, tok_options)
    )

    # load questions
    tf.logging.info('Load questions to predict..')
    questions = []
    answers = []
    with open(FLAGS.predict_file, 'r') as f:
        for line in f:
            questions.append(json.loads(line)['question'])
            a = json.loads(line)['answer']
            all_a = []
            for ans in a:
                if FLAGS.answer_regex:
                    all_a.append(ans)
                else:
                    all_a.append(normalize_answer(ans))

            answers.append(all_a)

    tf.logging.info("Loaded {} questions".format(len(questions)))

    # TODO move this to tf.flags
    BATCH_SIZE = 500
    PREDS_PER_SPAN = 3
    N_DOCS = 10
    NUM_CANDIDATES = 40

    batches = [questions[i: i + BATCH_SIZE]
               for i in range(0, len(questions), BATCH_SIZE)]

    for bn, batch in enumerate(batches):
        tf.logging.info("Running prediction for batch {}".format(bn))

        eval_writer = FeatureWriter(
            filename=os.path.join(FLAGS.output_dir, "eval.tf_record"),
            is_training=False)
        eval_features = []
        eval_examples = []

        # retrieve top-n questions
        tf.logging.info('rank documents..')
        ranked_docs = ranker.batch_closest_docs(batch, k=N_DOCS, num_workers=5)
        all_docids, all_doc_scores = zip(*ranked_docs)

        # Flatten document ids and retrieve text from database.
        # We remove duplicates for processing efficiency.
        tf.logging.info('fetch documents..')
        flat_docids = list({d for docids in all_docids for d in docids})  # all unique doc ids
        did2didx = {did: didx for didx, did in enumerate(flat_docids)}  # doc ids to numerical doc ids
        doc_texts = processes.map(fetch_text, flat_docids)

        # Split and flatten documents. Maintain a mapping from doc (index in
        # flat list) to split (index in flat list).
        flat_splits = []
        didx2sidx = []
        tf.logging.info('split documents..')
        for text in doc_texts:
            splits = split_doc(text)
            didx2sidx.append([len(flat_splits), -1])
            for split in splits:
                flat_splits.append(split)
            didx2sidx[-1][1] = len(flat_splits)

        tf.logging.info('tokenize questions and paragraphs.. ')
        q_tokens = processes.map_async(tokenize_question, batch)
        s_tokens = processes.map_async(tokenize_document, flat_splits)
        q_tokens = q_tokens.get()
        s_tokens = s_tokens.get()

        tf.logging.info("Build inputs for BERT.. ")
        unique_id = 1000000000
        ex_index = -1
        for (q_index, q) in enumerate(batch):
            query_tokens = q_tokens[q_index]
            if len(query_tokens) == 0:
                continue
            for doc_position, did in enumerate(all_docids[q_index]):
                start, end = didx2sidx[did2didx[did]]
                for sidx in range(start, end):
                    para_tokens = s_tokens[sidx][0]
                    para_tok_to_ind = s_tokens[sidx][1]
                    if len(para_tokens) == 0:
                        continue
                    max_tokens_for_doc = FLAGS.max_seq_length - len(query_tokens) - 3
                    doc_spans = []
                    start_offset = 0
                    _DocSpan = collections.namedtuple(  # pylint: disable=invalid-name
                        "DocSpan", ["start", "length"])
                    while start_offset < len(para_tokens):
                        length = len(para_tokens) - start_offset
                        if length > max_tokens_for_doc:
                            length = max_tokens_for_doc
                        doc_spans.append(_DocSpan(start=start_offset, length=length))
                        if start_offset + length == len(para_tokens):
                            break
                        start_offset += min(length, FLAGS.doc_stride)

                    eval_examples.append({'qid': q_index, 'doc_tokens': s_tokens[sidx][2],
                                          'doc_pos': doc_position, 'doc_score': all_doc_scores[q_index][doc_position],
                                          'doc_id': did, 'sidx': sidx, 'span_lenght': len(para_tokens)})
                    ex_index += 1

                    for (doc_span_index, doc_span) in enumerate(doc_spans):
                        tokens = []
                        token_to_orig_map = {}
                        token_is_max_context = {}
                        segment_ids = []
                        tokens.append("[CLS]")
                        segment_ids.append(0)
                        for token in query_tokens:
                            tokens.append(token)
                            segment_ids.append(0)
                        tokens.append("[SEP]")
                        segment_ids.append(0)

                        for i in range(doc_span.length):
                            split_token_index = doc_span.start + i
                            token_to_orig_map[len(tokens)] = para_tok_to_ind[split_token_index]

                            is_max_context = check_is_max_context(doc_spans, doc_span_index, split_token_index)
                            token_is_max_context[len(tokens)] = is_max_context
                            tokens.append(para_tokens[split_token_index])
                            segment_ids.append(1)

                        tokens.append("[SEP]")
                        segment_ids.append(1)
                        input_ids = processes.map(convert_to_id, [tokens])[0]

                        # The mask has 1 for real tokens and 0 for padding tokens. Only real
                        # tokens are attended to.
                        input_mask = [1] * len(input_ids)

                        # Zero-pad up to the sequence length.
                        while len(input_ids) < FLAGS.max_seq_length:
                            input_ids.append(0)
                            input_mask.append(0)
                            segment_ids.append(0)

                        assert len(input_ids) == FLAGS.max_seq_length
                        assert len(input_mask) == FLAGS.max_seq_length
                        assert len(segment_ids) == FLAGS.max_seq_length

                        feature = InputFeatures(
                            unique_id=unique_id,
                            example_index=ex_index,
                            doc_span_index=doc_span_index,
                            tokens=tokens,
                            token_to_orig_map=token_to_orig_map,
                            token_is_max_context=token_is_max_context,
                            input_ids=input_ids,
                            input_mask=input_mask,
                            segment_ids=segment_ids)
                        # if ex_index % 700 == 20:
                        #     tf.logging.info("*** Example ***")
                        #     tf.logging.info("unique_id: %s" % (unique_id))
                        #     tf.logging.info("question_id: %s" % (eval_examples[ex_index]["qid"]))
                        #     tf.logging.info("example_index: %s" % (ex_index))
                        #     tf.logging.info("doc_span_index: %s" % (doc_span_index))
                        #     tf.logging.info("tokens: %s" % " ".join(
                        #         [tokenization.printable_text(x) for x in tokens]))
                        #     tf.logging.info("token_to_orig_map: %s" % " ".join(
                        #         ["%d:%d" % (x, y) for (x, y) in six.iteritems(token_to_orig_map)]))
                        #     tf.logging.info("token_is_max_context: %s" % " ".join([
                        #         "%d:%s" % (x, y) for (x, y) in six.iteritems(token_is_max_context)
                        #     ]))
                        #     tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                        #     tf.logging.info(
                        #         "input_mask: %s" % " ".join([str(x) for x in input_mask]))
                        #     tf.logging.info(
                        #         "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))

                        # add feature
                        eval_features.append(feature)
                        eval_writer.process_feature(feature)
                        unique_id += 1
        # del s_tokens
        # del q_tokens
        eval_writer.close()

        tf.logging.info("***** Running predictions *****")
        tf.logging.info("  Num orig examples = %d", len(eval_examples))
        tf.logging.info("  Num split examples = %d", len(eval_features))
        tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size)

        all_results = []

        predict_input_fn = input_fn_builder(
            input_file=eval_writer.filename,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=False)

        # If running eval on the TPU, you will need to specify the number of
        # steps.
        all_results = []
        for result in estimator.predict(
                predict_input_fn, yield_single_examples=True):
            if len(all_results) % 1000 == 0:
                tf.logging.info("Processing example: %d" % (len(all_results)))
            unique_id = int(result["unique_ids"])
            start_logits = [float(x) for x in result["start_logits"].flat]
            end_logits = [float(x) for x in result["end_logits"].flat]
            all_results.append(
                RawResult(
                    unique_id=unique_id,
                    start_logits=start_logits,
                    end_logits=end_logits))

        output_prediction_file = os.path.join(FLAGS.output_dir, "predictions.json")
        output_nbest_file = os.path.join(FLAGS.output_dir, "nbest_predictions.json")

        _, preds = write_predictions(eval_examples, eval_features, all_results,
                                     FLAGS.n_best_size, FLAGS.max_answer_length,
                                     FLAGS.do_lower_case, output_prediction_file,
                                     output_nbest_file, FLAGS)

        tf.logging.info("sort best answers to each question.. ")

        all_preds_for_q = {}
        for ex_index in range(len(eval_examples)):
            preds_ = preds[ex_index]
            for s in preds_[0:min(len(preds_), PREDS_PER_SPAN)]:
                if s['text'] == 'empty':
                    continue
                qid = eval_examples[ex_index]['qid']

                if qid not in all_preds_for_q:
                    all_preds_for_q[qid] = []

                s['text'] = normalize_answer(s['text'])
                s['doc_score'] = eval_examples[ex_index]['doc_score']
                s['doc_pos'] = eval_examples[ex_index]['doc_pos']
                # s['dids'] = [eval_examples[ex_index]['doc_id']] # TODO
                s['in_doc_pos'] = 0
                ptokens = s_tokens[eval_examples[ex_index]['sidx']][0]
                qtoken = q_tokens[qid]
                s['paragraph_score'] = get_jaccard_sim(ptokens, qtoken)
                s['context_len'] = eval_examples[ex_index]['span_lenght']
                s['span_len'] = len(s['text'].split(' '))
                all_preds_for_q[qid].append(s)

        out = []
        top_performance_batch = 0
        any_performance_batch = 0
        for qid in range(len(batch)):
            sorted_answers = sorted(all_preds_for_q[qid], key=lambda k: k['span_score'], reverse=True)
            sample_agg_ind = {}
            sample_agg = []

            for i, s in enumerate(sorted_answers):

                if s['text'] in sample_agg_ind:
                    s_ = sample_agg[sample_agg_ind[s['text']]]
                    s_['num_occ'] += 1
                    n = s_['num_occ']
                    s_['avg_span_score'] = s_['avg_span_score'] * (n - 1) / n + s['span_score'] / n
                    s_['max_span_score'] = max(s_['max_span_score'], s['span_score'])
                    s_['min_span_score'] = min(s_['min_span_score'], s['span_score'])
                    s_['avg_doc_score'] = s_['avg_doc_score'] * (n - 1) / n + s['doc_score'] / n
                    s_['max_doc_score'] = max(s['doc_score'], s_['max_doc_score'])
                    s_['min_doc_score'] = min(s['doc_score'], s_['min_doc_score'])
                    s_['avg_doc_pos'] = s_['avg_doc_pos'] * (n - 1) / n + s['doc_pos'] / n
                    s_['max_doc_pos'] = max(s['doc_pos'], s_['max_doc_pos'])
                    s_['min_doc_pos'] = min(s['doc_pos'], s_['min_doc_pos'])
                    s_['avg_in_doc_pos'] = s_['avg_in_doc_pos'] * (n - 1) / n + s['in_doc_pos'] / n
                    s_['max_in_doc_pos'] = max(s['in_doc_pos'], s_['max_in_doc_pos'])
                    s_['min_in_doc_pos'] = min(s['in_doc_pos'], s_['min_in_doc_pos'])
                    s_['avg_paragraph_score'] = s_['avg_paragraph_score'] * (n - 1) / n + s['paragraph_score'] / n
                    s_['max_paragraph_score'] = max(s['paragraph_score'], s_['max_paragraph_score'])
                    s_['min_paragraph_score'] = min(s['paragraph_score'], s_['min_paragraph_score'])
                    s_['avg_end_logit'] = s_['avg_end_logit'] * (n - 1) / n + s['end_logit'] / n
                    s_['max_end_logit'] = max(s['end_logit'], s_['max_end_logit'])
                    s_['min_end_logit'] = min(s['end_logit'], s_['min_end_logit'])
                    s_['avg_start_logit'] = s_['avg_start_logit'] * (n - 1) / n + s['start_logit'] / n
                    s_['max_start_logit'] = max(s['start_logit'], s_['max_start_logit'])
                    s_['min_start_logit'] = min(s['start_logit'], s_['min_start_logit'])

                else:
                    s['first_occ'] = i + 1
                    s['num_occ'] = 1
                    s['avg_span_score'] = s['span_score']
                    s['max_span_score'] = s['span_score']
                    s['min_span_score'] = s['span_score']
                    s['avg_doc_score'] = s['doc_score']
                    s['max_doc_score'] = s['doc_score']
                    s['min_doc_score'] = s['doc_score']
                    s['avg_doc_pos'] = s['doc_pos']
                    s['max_doc_pos'] = s['doc_pos']
                    s['min_doc_pos'] = s['doc_pos']
                    s['avg_start_logit'] = s['start_logit']
                    s['max_start_logit'] = s['start_logit']
                    s['min_start_logit'] = s['start_logit']
                    s['avg_end_logit'] = s['end_logit']
                    s['max_end_logit'] = s['end_logit']
                    s['min_end_logit'] = s['end_logit']
                    s['avg_paragraph_score'] = s['paragraph_score']
                    s['max_paragraph_score'] = s['paragraph_score']
                    s['min_paragraph_score'] = s['paragraph_score']
                    s['avg_in_doc_pos'] = s['in_doc_pos']
                    s['max_in_doc_pos'] = s['in_doc_pos']
                    s['min_in_doc_pos'] = s['in_doc_pos']
                    s['target'] = 0
                    s['question'] = questions[bn * BATCH_SIZE + qid]
                    s['question_len'] = len(questions[bn * BATCH_SIZE + qid].split(' '))
                    for ans in answers[bn * BATCH_SIZE + qid]:
                        if FLAGS.answer_regex:
                            try:
                                compiled = re.compile(ans, flags=re.IGNORECASE + re.UNICODE + re.MULTILINE)
                            except BaseException:
                                tf.logging.warn('Regular expression failed to compile: %s' % ans)
                                continue
                            if compiled.match(s['text']) is not None:
                                s['target'] = 1
                                break
                        elif not FLAGS.answer_regex and s['text'] == ans:
                            s['target'] = 1
                            break
                    sample_agg_ind[s['text']] = len(sample_agg)
                    sample_agg.append(s)

                if len(sample_agg_ind) > NUM_CANDIDATES:
                    break

            out.append(sample_agg)
            top_performance_batch += sample_agg[0]['target']
            found = False
            for s in sample_agg:
                if s['target'] == 1:
                    found = True
                    break
            any_performance_batch += (1 if found else 0)

        print(top_performance_batch / BATCH_SIZE)
        print(any_performance_batch / BATCH_SIZE)
        with open(os.path.join('out', '{}-feat-batch-{}.txt'.format(FLAGS.out_name, bn)), 'w') as f:
            for o in out:
                f.write(json.dumps(o) + '\n')


if __name__ == "__main__":
    flags.mark_flag_as_required("vocab_file")
    flags.mark_flag_as_required("bert_config_file")
    flags.mark_flag_as_required("output_dir")
    tf.app.run()
