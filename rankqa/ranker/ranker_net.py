import logging

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from .model import RankNetModel

logger = logging.getLogger(__name__)


class RankerNet(object):

    def __init__(self, args, num_feat):
        self.args = args
        self.network = RankNetModel(args, num_feat)
        if self.args.cuda:
            self.network.cuda()
        self.optimizer = optim.Adam(self.network.parameters(), lr=args.lr)
        self.loss_func = nn.functional.mse_loss
        self.loss_func_single = nn.functional.mse_loss

    def predict(self, input):
        self.network.eval()
        if self.args.cuda:
            input = Variable(input.cuda(async=True))
        else:
            input = Variable(input)
        scores = self.network.predict(input)
        return scores.data.cpu()

    def eval_pairwise(self, input_l, input_r, targets):
        self.network.eval()
        with torch.no_grad():
            if self.args.cuda:
                targets = Variable(targets.cuda(async=True))
                input_l = Variable(input_l.cuda(async=True))
                input_r = Variable(input_r.cuda(async=True))
            else:
                targets = Variable(targets)
                input_l = Variable(input_l)
                input_r = Variable(input_r)

            y_pred = self.network.forward_pairwise(input_l, input_r)

            loss = self.loss_func_single(y_pred[:, 0], targets)
        return loss.item()

    def update_pairwise(self, input_l, input_r, targets):
        self.network.train()

        self.network.zero_grad()
        if self.args.cuda:
            targets = Variable(targets.cuda(async=True))
            input_l = Variable(input_l.cuda(async=True))
            input_r = Variable(input_r.cuda(async=True))
        else:
            targets = Variable(targets)
            input_l = Variable(input_l)
            input_r = Variable(input_r)

        y_pred = self.network.forward_pairwise(input_l, input_r)

        loss = self.loss_func_single(y_pred[:, 0], targets)
        #
        l2_reg = None
        for W in self.network.parameters():
            if l2_reg is None:
                l2_reg = W.norm(2)
            else:
                l2_reg = l2_reg + W.norm(2)
        loss = loss + self.args.reg * l2_reg  # todo args

        loss.backward()
        self.optimizer.step()
        return loss.item()

    def safe(self, path):
        torch.save(self.network, path)
        pass

    def load(self, path):
        self.network = torch.load(path)
        pass
