from math import pi
from namedtensor import ntorch


class PlaceCells:
    """Cells to provide ground truth activations as train and test output"""

    def __init__(self, scale, n, scene):
        place_cells = scene.random(n)
        self.centers = ntorch.tensor(place_cells, names=("placecell", "ax"))
        self.scale = scale

    def __call__(self, x):
        z = -(self.centers - x).pow(2).sum("ax") / (2 * self.scale ** 2)
        return z.softmax("placecell")


class HeadDirectionCells:
    """Cells to provide ground truth activations as train and test output"""

    def __init__(self, concentration, n):
        self.centers = ntorch.randn(n, names=("hdcell")) * 2 * pi - pi
        self.k = concentration

    def __call__(self, phi):
        z = self.k * (phi - self.centers).cos()
        return z.softmax("hdcell")
