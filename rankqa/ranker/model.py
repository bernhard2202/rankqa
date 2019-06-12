import torch.nn as nn


class RankNetModel(nn.Module):

    def __init__(self, args, feat_size):
        super(RankNetModel, self).__init__()

        self.linear = nn.Linear(feat_size, args.linear_dim)
        self.activ2 = nn.ReLU()
        self.linear2 = nn.Linear(int(args.linear_dim), 1)

        self.output_sig = nn.Sigmoid()

    def forward(self, inputl):
        return self.output_sig(self._forward_pass(inputl))

    def forward_pairwise(self, input1, input2):
        s1 = self._forward_pass(input1)
        s2 = self._forward_pass(input2)

        out = self.output_sig(s1 - s2)
        return out

    def _forward_pass(self, input_sample):
        out = self.linear(input_sample)
        out = self.activ2(out)
        out = self.linear2(out)

        return out

    def predict(self, input):
        return self._forward_pass(input)
