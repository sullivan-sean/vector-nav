from math import pi
import matplotlib.pyplot as plt
from namedtensor import ntorch
import torch
import numpy as np
from scene import SquareCage


class PlaceCells:
    """Cells to provide ground truth activations as train and test output"""

    def __init__(self, scale, n, scene, seed=None):
        if isinstance(scene, SquareCage):
            rs = np.random.RandomState(seed)
            place_cells = rs.uniform(-scene.height / 2, scene.height / 2, size=(n, 2))
        else:
            place_cells = scene.random(n)
        self.centers = ntorch.tensor(place_cells, names=("placecell", "ax")).float().cuda()
        self.scale = scale

        plt.scatter(*place_cells.T)
        plt.show()

    def unnor_logpdf(self, x):
        return -(self.centers - x).pow(2).sum("ax") / (2 * self.scale ** 2)

    def __call__(self, x):
        logp = self.unnor_logpdf(x)
        res = logp - logp.logsumexp('placecell')
        return res.softmax('placecell')


class HeadDirectionCells:
    """Cells to provide ground truth activations as train and test output"""

    def __init__(self, concentration, n, seed=None):
        rs = np.random.RandomState(seed)
        centers = rs.uniform(-np.pi, np.pi, size=(n))
        self.centers = ntorch.tensor(centers, names=("hdcell")).float().cuda()
        self.k = concentration

    def unnor_logpdf(self, phi):
        return self.k * (phi - self.centers).cos()

    def __call__(self, x):
        logp = self.unnor_logpdf(x)
        res = logp - logp.logsumexp('hdcell')
        return res.softmax('hdcell')
