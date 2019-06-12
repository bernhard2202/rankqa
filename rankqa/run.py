import argparse
import logging
import os
import sys
import time
import uuid

import numpy as np
import torch
from evaluate import Evaluator
from ranker import data_utils, data_loader
from ranker import ranker_net
from ranker.data_utils import batchify_pair

np.random.seed(42)
torch.manual_seed(1)
torch.cuda.manual_seed(1)

logger = logging.getLogger()

# ---------------------------------------------------------------------------------------------
# description of features to use during training
# ---------------------------------------------------------------------------------------------

feature_descriptors = [
    {'feature_name': 'sum_span_score', 'normscheme': 'minmax', 'preprocess': np.log},
    {'feature_name': 'sum_doc_score', 'normscheme': 'minmax', 'preprocess': np.log},
    {'feature_name': 'doc_score', 'normscheme': 'minmax', 'preprocess': np.log},
    {'feature_name': 'span_score', 'normscheme': 'minmax', 'preprocess': np.log},
    {'feature_name': 'min_doc_score', 'normscheme': 'minmax', 'preprocess': np.log},
    {'feature_name': 'max_doc_score', 'normscheme': 'minmax', 'preprocess': np.log},
    {'feature_name': 'avg_doc_score', 'normscheme': 'minmax', 'preprocess': np.log},
    {'feature_name': 'max_span_score', 'normscheme': 'minmax', 'preprocess': np.log},
    {'feature_name': 'min_span_score', 'normscheme': 'minmax', 'preprocess': np.log},
    {'feature_name': 'avg_span_score', 'normscheme': 'minmax', 'preprocess': np.log},
    {'feature_name': 'first_occ', 'normscheme': 'minmax', 'preprocess': np.log},
    {'feature_name': 'num_occ', 'normscheme': 'minmax', 'preprocess': np.log},
    {'feature_name': 'context_len', 'normscheme': 'minmax', 'preprocess': np.log},
    {'feature_name': 'question_len', 'normscheme': 'minmax', 'preprocess': np.log}
]

train_filesSQ = [os.path.join('features',
                              'SQuAD-v1.1-train-default-pipeline.preds-part-{}'.format(i)) for i in range(88)]

train_filesWM = [os.path.join('features',
                              'WikiMovies-train-default-pipeline.preds-part-{}'.format(i)) for i in range(96)]

train_filesWQ = [os.path.join('features',
                              'WebQuestions-train-default-pipeline.preds-part-{}'.format(i)) for i in range(4)]

train_filesT = [os.path.join('features',
                             'CuratedTrec-train-default-pipeline.preds-part-{}'.format(i)) for i in range(2)]

test_files = {

    'SQuAD': [os.path.join('features',
                           'SQuAD-v1.1-dev-default-pipeline.preds-part-{}'.format(i)) for i in range(11)],

    'WikiMovies': [os.path.join('features',
                                'WikiMovies-test-default-pipeline.preds-part-{}'.format(i)) for i in range(10)],

    'CuratedTREC': [os.path.join('features',
                                 'CuratedTrec-test-default-pipeline.preds-part-{}'.format(i)) for i in range(1)],

    'WebQuestions': [os.path.join('features',
                                  'WebQuestions-test-default-pipeline.preds-part-{}'.format(i)) for i in range(3)],
}

train_files = [train_filesSQ, train_filesWM, train_filesWQ, train_filesT]


# -------------------
# Arguments
# -------------------


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser()
# subsampling
parser.add_argument('--max-per-q', type=int, default=2, help='Maximum number of training pairs that are sampled'
                                                             ' per question')
parser.add_argument('--valid-split', type=float, default=0.9, help='amount to use for training vs. model selection')
parser.add_argument('--stratify-valid', type=str2bool, default=True, help='make validation data of equal size for all'
                                                                          ' data sets')
parser.add_argument('--max-depth', type=int, default=2, help='maximally traverse this deep through candidate answers '
                                                             ' when sampling training pairs')

# fixed stuff
parser.add_argument('--cuda', type=str2bool, default=True, help='Enable CUDA support, ie. run on gpu')

# learning
parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')
parser.add_argument('--epochs', type=int, default=100, help='max. number of epochs')
parser.add_argument('--batch-size', type=int, default=256, help='training batch size')

# other
parser.add_argument('--name', type=str, default='regular')

# tuning
parser.add_argument('--linear-dim', type=int, default=512)
parser.add_argument('--reg', type=float, default=0.00005, help='regularization constant for weight penalty')

args = parser.parse_args()


def val(valid_loader_, model_, args_):
    valid_loss = []
    for i, input in enumerate(valid_loader_):
        inl, inr, target = input
        l = model_.eval_pairwise(inl, inr, target)
        valid_loss.append(l)
    return np.mean(valid_loss)


def train(train_loader_, valid_loader_, model_, args_, evaluators_, modelfilename_):
    initiallywrong = []
    for evaluator_ in evaluators_:
        log, wrongansw = evaluator_.evaluate(model_)
        initiallywrong.append(wrongansw)
        logger.info(log)
    best_val_loss = float('inf')
    best_val_iteration = 0

    for b in range(args_.epochs):
        loss = []
        logger.info('==================== EPOCH {} ================================'.format(b))
        for i, input in enumerate(train_loader_):
            inl, inr, target = input
            l = model_.update_pairwise(inl, inr, target)
            loss.append(l)
        val_loss = val(valid_loader_, model_, args_)
        logger.info('Epoch finished avg loss = {}'.format(np.mean(loss)))
        logger.info('Validation loss = {}'.format(val_loss))

        # check if
        if best_val_loss > val_loss:
            # save model
            logger.info('BEST EPOCH SO FAR --> safe model')
            model_.safe(modelfilename_)
            best_val_loss = val_loss
            best_val_iteration = 0
        best_val_iteration += 1
        if best_val_iteration > 10:
            # stop training
            logger.info("EARLY STOPPING")
            break
    model_.load(modelfilename_)

    # evaluate
    for i, evaluator_ in enumerate(evaluators_):
        log, wrong = evaluator_.evaluate(model_)
        logger.info(log)
        logger.info('fraction still correct = {}'.format(len(np.intersect1d(initiallywrong[i], wrong)) /
                                                         len(wrong)))


if __name__ == "__main__":
    # Prepare logging
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)
    ts = time.strftime("%Y%m%d-") + str(uuid.uuid4())[:8]

    logfilename = os.path.join('out', args.name + ts + ".txt")
    modelfilename = os.path.join('out', args.name + ts + ".mdl")
    logfile = logging.FileHandler(logfilename, 'w')
    logfile.setFormatter(fmt)
    logger.addHandler(logfile)
    logger.info('COMMAND: %s' % ' '.join(sys.argv))

    logger.info(args)

    # load training data
    logger.info('Loading {} files with training samples..'.format(sum(len(x) for x in train_files)))
    train_data, valid_data, normalizers = data_utils.load_subsample(train_files, feature_descriptors, args)
    logger.info('Done. Number of train pairs loaded: {} (valid = {})'.format(len(train_data), len(valid_data)))

    # load dev data
    evaluators = []
    for dataset in test_files:
        logger.info('Loading {} dataset with {} files..'.format(dataset, len(test_files[dataset])))
        test_data = data_utils.load_full(test_files[dataset])
        test_dataset = data_utils.build_validation_dataset(test_data, normalizers)
        evaluator = Evaluator(test_dataset[0], test_dataset[1], dataset)
        evaluators.append(evaluator)

    # generate train data loader
    logger.info('Initialize training loader..')
    train_dataset = data_loader.PairwiseRankingDataSet(train_data, normalizers)
    valid_dataset = data_loader.PairwiseRankingDataSet(valid_data, normalizers)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.RandomSampler(train_dataset),
        pin_memory=args.cuda,
        collate_fn=batchify_pair
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.RandomSampler(valid_dataset),
        pin_memory=args.cuda,
        collate_fn=batchify_pair
    )

    # init model
    logger.info('Init model..')
    model = ranker_net.RankerNet(args, train_dataset.num_feat)
    logger.info('Done.')

    # kick off training
    train(train_loader, valid_loader, model, args, evaluators, modelfilename)
    for handler in logger.handlers:
        handler.flush()
