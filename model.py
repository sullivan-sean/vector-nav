from namedtensor import ntorch


class VecNavModel(ntorch.nn.Module):
    """LSTM supervised model from DeepMind paper in namedtensor"""

    def __init__(self, head_cells, place_cells, hidden_size=128, g_size=512):
        super(VecNavModel, self).__init__()
        self.rnn = ntorch.nn.LSTM(3, hidden_size).spec("input", "t", "hidden")
        self.drop = ntorch.nn.Dropout(0.5)
        self.init_state = ntorch.nn.Linear(
            head_cells + place_cells, hidden_size
        ).spec("cells", "hidden")
        self.init_cell = ntorch.nn.Linear(
            head_cells + place_cells, hidden_size
        ).spec("cells", "hidden")
        self.g = ntorch.nn.Linear(hidden_size, g_size).spec("hidden", "g")
        self.head = ntorch.nn.Linear(g_size, head_cells).spec("g", "hdcell")
        self.place = ntorch.nn.Linear(g_size, place_cells).spec(
            "g", "placecell"
        )

    def forward(self, seq, c0, h0):
        c0 = c0.rename("placecell", "cells")
        h0 = h0.rename("hdcell", "cells")

        cells = ntorch.cat([h0, c0], "cells")

        l0 = self.init_state(cells)
        m0 = self.init_cell(cells)

        out, _ = self.rnn(seq, (m0, l0))

        g = self.drop(self.g(out))

        z, y = self.head(g), self.place(g)

        return z.log_softmax("hdcell"), y.log_softmax("placecell"), g


class CrossEntropyLoss:
    """Define custom CrossEntropyLoss for soft loss instead of PyTorch hard"""

    def __init__(self):
        self.name = None

    def spec(self, name):
        # Make loss compatible with namedtensor
        self.name = name
        return self

    def __call__(self, pred, targets):
        return -1 * (targets * pred).sum(self.name).mean().values
