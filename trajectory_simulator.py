import matplotlib.pyplot as plt
from namedtensor import ntorch
import numpy as np
import torch
from util import unit_vector, rotation_matrix


class TrajectorySimulator:
    def __init__(self, scene, **params):
        self.params = {
            "T": 15,
            "std_dev_forward": 0.13,
            "std_dev_rotation": np.pi * 330 / 180,
            "mean_rotation": 0,
            "v0": 0.20,
            "perimeter": 0.03,
            "velocity_reduction": 0.25,
            "trajectory_length": 100,
        }
        self.scene = scene
        self.set_params(**params)

    def set_params(self, **params):
        self.params = {**self.params, **params}

    @property
    def attrs(self):
        return self.params

    def _check_attrs(self, file):
        attrs = file.attrs
        try:
            for k, v in self.attrs.items():
                assert (attrs[k] == v).all()
        except AssertionError:
            raise Exception(
                "Trying to modify a file with different params, aborting"
            )

    def save(self, file, N=100):
        if not file.empty:
            self._check_attrs(file)
        file.save_data([i.values.numpy() for i in self.trajectories(N=N)])
        file.set_attrs(self.attrs)

    def load_params(self, params):
        self.set_params(**params)

    def trajectories(self, N=100, dt=0.02):
        perimeter = self.params['perimeter']
        T = self.params["T"]
        n = int(T / dt)
        mu, sigma, b = [
            self.params[i]
            for i in ["mean_rotation", "std_dev_rotation", "std_dev_forward"]
        ]

        rotation_velocities = torch.tensor(
            np.random.normal(mu, sigma, size=(n, N))
        ).float()
        forward_velocities = torch.tensor(
            np.random.rayleigh(b, size=(n, N))
        ).float()

        positions = ntorch.zeros((n, 2, N), names=("t", "ax", "sample"))
        vs = torch.zeros((n, N))
        angles = rotation_velocities
        directions = torch.zeros((n, 2, N))

        vs[0] = self.params["v0"]
        theta = torch.rand(N) * 2 * np.pi
        directions[0] = unit_vector(theta)
        positions[{"t": 0}] = ntorch.tensor(
            self.scene.random(N), names=("sample", "ax")
        )

        for i in range(1, n):
            dist, phi = self.scene.closestWall(
                positions[{"t": i - 1}].values, directions[i - 1]
            )
            wall = (dist < perimeter) & (phi.abs() < np.pi / 2)
            angle_correction = torch.where(
                wall,
                phi.sign() * (np.pi / 2 - phi.abs()),
                torch.zeros_like(phi)
            )
            angles[i] += angle_correction

            vs[i] = torch.where(
                wall,
                (1 - self.params["velocity_reduction"]) * (vs[i - 1]),
                forward_velocities[i],
            )
            positions[{"t": i}] = (
                positions[{"t": i - 1}] + directions[i - 1] * vs[i] * dt
            )

            mat = rotation_matrix(angles[i] * dt)
            directions[i] = torch.einsum("ijk,jk->ik", mat, directions[i - 1])

        idx = np.round(
            np.linspace(0, n - 2, self.params["trajectory_length"])
        ).astype(int)
        # idx = np.array(sorted(np.random.choice(np.arange(n), size=self.params["trajectory_length"], replace=False)))

        dphis = ntorch.tensor(angles[idx] * dt, names=("t", "sample"))
        velocities = ntorch.tensor(vs[idx], names=("t", "sample"))
        vel = ntorch.stack((velocities, dphis.cos(), dphis.sin()), "input")

        xs = ntorch.tensor(
            positions.values[idx], names=("t", "ax", "sample")
        )
        # xs0 = positions[{'t': 0}]
        xs0 = ntorch.tensor(self.scene.random(n=N), names=("sample", "ax"))

        hd = torch.atan2(directions[:, 1], directions[:, 0])
        hd0 = ntorch.tensor(hd[0][None], names=("hd", "sample"))
        hd = ntorch.tensor(hd[idx + 1][None], names=("hd", "t", "sample"))

        xs = xs.transpose('sample', 't', 'ax')
        hd = hd.transpose('sample', 't', 'hd')
        vel = vel.transpose('sample', 't', 'input')
        xs0 = xs0.transpose('sample', 'ax')
        hd0 = hd0.transpose('sample', 'hd')

        return xs, hd, vel, xs0, hd0

    def plot_trajectories(self, *args, **kwargs):
        self.scene.plot()
        ps = self.trajectories(*args, **kwargs)[0]
        for t in ps.values.transpose(1, 2):
            x, y = t.numpy()
            plt.plot(x, y)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()

