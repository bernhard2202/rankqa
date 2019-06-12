import numpy as np
import torch
from tqdm import tqdm


class Evaluator(object):

    def __init__(self, X, y, name):
        self.y = y
        self.name = name
        self.X = X
        self.max = 0

    def evaluate(self, model):
        baseline_performance = 0
        top_performance = 0
        model_performance = 0
        wrong = []
        with torch.no_grad():
            for i, x in tqdm(enumerate(self.X)):
                inputs = []
                solvable = False
                for j, candidate in enumerate(x):
                    if self.y[i][j] == 1:
                        solvable = True
                    inputs.append(candidate)

                baseline_performance += int(self.y[i][0])

                if not solvable:
                    # no need to predict unsolvable questions
                    wrong.append(i)
                    continue

                top_performance += 1

                scores = model.predict(torch.stack(inputs))

                model_performance += int(self.y[i][np.argmax(scores[0:10])])
                if int(self.y[i][np.argmax(scores)]) == 0:
                    wrong.append(i)

        baseline_performance = baseline_performance / len(self.y)
        model_performance = model_performance / len(self.y)
        top_performance = top_performance / len(self.y)
        log = '{} evaluation results w/o re-ranking: {} / upper bound: {}; rank-qa: {}'.format(self.name,
                                                                                               baseline_performance,
                                                                                               top_performance,
                                                                                               model_performance)
        return log, wrong
