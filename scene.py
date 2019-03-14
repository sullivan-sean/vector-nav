import matplotlib.pyplot as plt
import torch
from math import pi
from util import angle_between, unit_vector


class Scene:
    """Abstract scene class"""

    def __init__(self):
        pass

    def closestWall(self, pos, direction):
        """Give distance and angle to closest wall given position and dir"""
        raise NotImplementedError()

    def plot(self):
        """Plot the scene"""
        raise NotImplementedError()

    def random(self, n=1):
        """Return n random points within the scene boundaries"""
        raise NotImplementedError()


class CircularCage(Scene):
    def __init__(self, diameter):
        super(CircularCage, self).__init__()
        self.radius = diameter / 2

    def closestWall(self, position, direction):
        # closest wall on a circle lies on a line through the origin
        theta = torch.atan2(position[1], position[0]).float()
        wall = self.radius * unit_vector(theta)

        min_dist = torch.norm(wall.abs() - position.abs(), p=2, dim=0)

        angle = angle_between(position, direction)
        return min_dist, angle

    def plot(self):
        ts = torch.linspace(0, 2 * pi, 1000)
        data = unit_vector(ts).numpy() * self.radius
        plt.plot(*data)

    def random(self, n=1):
        t = 2 * pi * torch.rand(n).float()
        u = torch.rand((n, 2)).sum(dim=1)
        r = torch.where(u > 1, 2 - u, u)
        return (r * unit_vector(t)).transpose(0, 1) * self.radius


class RectangularCage(Scene):
    def __init__(self, height, width):
        super(RectangularCage, self).__init__()
        self.height = height
        self.width = width

    def closestWall(self, pos, direction):
        # Find closest wall to an edge in the rectangle
        dists = (
            torch.tensor([self.width / 2, self.height / 2]).reshape(2, 1)
            - pos.abs()
        )
        min_dist, ax = dists.min(dim=0)
        signs = pos[ax, torch.arange(pos.shape[1])] > 0
        angle = (3 - 2 * signs.float()) * pi / 2 - (1 - ax.float()) * pi / 2
        return min_dist, angle_between(unit_vector(angle), direction)

    def plot(self):
        ws = torch.linspace(-0.5, 0.5, 1000) * self.width
        hs = torch.linspace(-0.5, 0.5, 1000) * self.height

        w2s = torch.ones_like(hs) * self.width / 2
        h2s = torch.ones_like(ws) * self.height / 2
        xs = torch.cat([ws, w2s, ws.flip(0), -w2s]).numpy()
        ys = torch.cat([h2s, hs.flip(0), -h2s, hs]).numpy()
        plt.plot(xs, ys)

    def random(self, n=1):
        return (torch.rand((n, 2)) - 0.5) * torch.tensor(
            [[self.height, self.width]]
        ).float()


class SquareCage(RectangularCage):
    def __init__(self, height):
        super(SquareCage, self).__init__(height, height)
