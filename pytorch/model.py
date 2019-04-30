from namedtensor import ntorch
from torch.nn import functional as F
import numpy as np


def displaced_linear_initializer(tensor, input_size, displace):
    stddev = 1. / np.sqrt(input_size)
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(stddev).add_(stddev * displace)


class VecNavModel(ntorch.nn.Module):
    """LSTM supervised model from DeepMind paper in namedtensor"""

    def __init__(self,
                 head_cells,
                 place_cells,
                 hidden_size=128,
                 g_size=512,
                 g_bias=True):
        super(VecNavModel, self).__init__()
        self.rnn = ntorch.nn.LSTM(3, hidden_size).spec("input", "t", "hidden")
        self.init_state = ntorch.nn.Linear(head_cells + place_cells,
                                           hidden_size).spec(
                                               "cells", "hidden")
        self.init_cell = ntorch.nn.Linear(head_cells + place_cells,
                                          hidden_size).spec("cells", "hidden")
        self.g = ntorch.nn.Linear(
            hidden_size, g_size, bias=g_bias).spec("hidden", "g")
        self.head = ntorch.nn.Linear(g_size, head_cells).spec("g", "hdcell")
        self.place = ntorch.nn.Linear(g_size, place_cells).spec(
            "g", "placecell")

        for l in [self.head, self.place]:
            displaced_linear_initializer(l.weight, g_size, 0)

    def forward(self, seq, c0, h0):
        cells = ntorch.cat([h0, c0], ["hdcell", "placecell"], name="cells")
        initial_state = (self.init_cell(cells), self.init_state(cells))

        out, _ = self.rnn(seq, initial_state)

        g = F.dropout(self.g(out).transpose("batch", "g", "t").values, 0.5, self.training)
        g = ntorch.tensor(g, names=("batch", "g", "t"))

        return self.head(g), self.place(g), g


class CrossEntropyLoss:
    """Define custom CrossEntropyLoss for soft loss instead of PyTorch hard"""

    def __init__(self):
        self.name = None

    def spec(self, name):
        # Make loss compatible with namedtensor
        self.name = name
        return self

    def __call__(self, pred, targets):
        prod = targets * pred.log_softmax(self.name)
        return -(prod.sum(self.name).mean().values)
