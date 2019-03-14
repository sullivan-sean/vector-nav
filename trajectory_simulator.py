from cell_group import PlaceCells, HeadDirectionCells
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
            "std_dev_rotation": 2 * np.pi * 330 / 360,
            "mean_rotation": 0,
            "v0": 0.20,
            "perimeter": 0.03,
            "velocity_reduction": 0.75,
            "place_cell_scale": 0.01,
            "hd_concentration": 20,
            "place_cell_count": 256,
            "hd_cell_count": 12,
            "trajectory_length": 100,
        }
        self.scene = scene
        self.set_params(**params)
        self.place_cells = PlaceCells(
            self.params["place_cell_scale"],
            self.params["place_cell_count"],
            self.scene,
        )
        self.hd_cells = HeadDirectionCells(
            self.params["hd_concentration"], self.params["hd_cell_count"]
        )

    def set_params(self, **params):
        self.params = {**self.params, **params}

    @property
    def attrs(self):
        return {
            **self.params,
            "place_cell_centers": self.place_cells.centers.values.numpy(),
            "hd_cell_centers": self.hd_cells.centers.values.numpy(),
        }

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

        xs, dphis, v, phis = self.trajectories(N=N)

        cs = self.place_cells(xs)
        hs = self.hd_cells(phis)

        inp = ntorch.stack((v, dphis.cos(), dphis.sin()), "input")
        data = ntorch.cat(
            [
                obj.rename(d, "tmp")
                for obj, d in zip(
                    [xs, cs, hs, inp], ["ax", "placecell", "hdcell", "input"]
                )
            ],
            dim="tmp",
        )
        data = data.transpose("sample", "t", "tmp").values.numpy()

        file.save_data(data)
        file.set_attrs(self.attrs)

    def load_params(self, params):
        self.place_cells.centers = ntorch.tensor(
            params.pop("place_cell_centers"), names=("placecell", "ax")
        )
        self.hd_cells.centers = ntorch.tensor(
            params.pop("hd_cell_centers"), names=("hdcell")
        )
        self.set_params(**params)

    def trajectories(self, N=100, dt=0.02):
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
        angles = torch.zeros((n, N))
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
            wall = (dist < self.params["perimeter"]) & (phi.abs() < np.pi / 2)
            angle_correction = wall.float() * (
                phi.sign() * (np.pi / 2 - phi.abs())
            )
            angles[i] = angle_correction + rotation_velocities[i] * dt

            vs[i] = torch.where(
                wall,
                vs[i - 1]
                - self.params["velocity_reduction"] * (vs[i - 1] - 0.05),
                forward_velocities[i],
            )
            positions[{"t": i}] = (
                positions[{"t": i - 1}] + directions[i - 1] * vs[i] * dt
            )

            mat = rotation_matrix(angles[i])
            directions[i] = torch.einsum("ijk,jk->ik", mat, directions[i - 1])

        idx = idx = np.round(
            np.linspace(0, n - 1, self.params["trajectory_length"])
        ).astype(int)
        positions = ntorch.tensor(
            positions.values[idx], names=("t", "ax", "sample")
        )
        dphis = ntorch.tensor(angles[idx], names=("t", "sample"))
        velocities = ntorch.tensor(vs[idx], names=("t", "sample"))

        phis = torch.atan2(directions[:, 1], directions[:, 0])
        phis = ntorch.tensor(phis[idx], names=("t", "sample"))

        return positions, dphis, velocities, phis

    def plot_trajectories(self, *args, **kwargs):
        self.scene.plot()
        ps, _, _, _ = self.trajectories(*args, **kwargs)
        for t in ps.values.transpose(0, 2):
            x, y = t.numpy()
            plt.plot(x, y)
        plt.show()
