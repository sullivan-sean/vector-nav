import numpy as np
import tensorflow as tf


class CellEnsemble:
    def __init__(self, n_cells):
        self.n_cells = n_cells

    def get_targets(self, x):
        lp = self.logpdf(x)
        log_posteriors = lp - tf.reduce_logsumexp(lp, axis=2, keep_dims=True)
        return tf.nn.softmax(log_posteriors, dim=-1)

    def get_init(self, x):
        return self.get_targets(x)

    def loss(self, predictions, x, name="ensemble_loss"):
        labels = self.get_targets(x)
        return tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels,
                                                          logits=predictions,
                                                          name=name)


class PlaceCellEnsemble(CellEnsemble):
    def __init__(self, n_cells, stdev=0.35, env_size=10, seed=None):
        super(PlaceCellEnsemble, self).__init__(n_cells)
        pos_min = -env_size / 2.0
        pos_max = env_size / 2.0
        rs = np.random.RandomState(seed)
        self.means = rs.uniform(pos_min, pos_max, size=(self.n_cells, 2))
        self.variances = np.ones_like(self.means) * stdev**2

    def logpdf(self, trajs):
        diff = trajs[:, :, tf.newaxis, :] - self.means[np.newaxis, np.
                                                       newaxis, ...]
        return -0.5 * tf.reduce_sum((diff**2) / self.variances, axis=-1)


class HeadDirectionCellEnsemble(CellEnsemble):
    def __init__(self, n_cells, concentration=20, seed=None):
        super(HeadDirectionCellEnsemble, self).__init__(n_cells)
        # Create a random Von Mises with fixed cov over the position
        rs = np.random.RandomState(seed)
        self.means = rs.uniform(-np.pi, np.pi, (n_cells))
        self.kappa = np.ones_like(self.means) * concentration

    def logpdf(self, x):
        return self.kappa * tf.cos(x - self.means[np.newaxis, np.newaxis, :])
