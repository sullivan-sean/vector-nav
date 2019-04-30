import numpy as np
import sonnet as snt
import tensorflow as tf


def init_layer(input_size, displace, dtype=tf.float32):
    stddev = 1. / np.sqrt(input_size)
    return tf.truncated_normal_initializer(mean=displace * stddev,
                                           stddev=stddev,
                                           dtype=dtype)


class GridCellsRNN(snt.AbstractModule):
    """RNN computes place and head-direction cell predictions from velocities."""

    def __init__(self,
                 ens,
                 lstm_size,
                 g_size,
                 g_has_bias=False,
                 name="grid_cell_supervised"):
        super(GridCellsRNN, self).__init__(name=name)
        self._g_size = g_size
        self._g_has_bias = g_has_bias
        self._lstm_size = lstm_size
        self._ens = ens
        self.training = False

        self._core = tf.keras.layers.LSTMCell(self._lstm_size)

    def _build(self, init_conds, vels, training=False):
        init_conds = tf.concat(init_conds, axis=1)

        init_state = snt.Linear(self._lstm_size, name="state_init")(init_conds)
        init_cell = snt.Linear(self._lstm_size, name="cell_init")(init_conds)

        self.training = training

        batch_size = vels.shape[0]
        initial_state = [init_state, init_cell]

        # Run LSTM
        lstm_output, final_state = tf.nn.dynamic_rnn(
            cell=self._core,
            inputs=vels,
            time_major=False,
            initial_state=initial_state)

        lstm_output = tf.reshape(lstm_output, shape=[-1, self._lstm_size])

        g = snt.Linear(
            self._g_size,
            use_bias=self._g_has_bias,
            regularizers={"w": tf.contrib.layers.l2_regularizer(1e-5)},
            name="bottleneck")(lstm_output)

        if self.training:
            tf.logging.info("Adding dropout layers")
            g = tf.nn.dropout(g, 0.5, name="dropout")

        ens = [
            snt.Linear(
                ens.n_cells,
                regularizers={"w": tf.contrib.layers.l2_regularizer(1e-5)},
                initializers={"w": init_layer(self._g_size, 0)},
                name="pc_logits")(g) for ens in self._ens
        ]

        ens = [
            tf.reshape(e, shape=[batch_size, -1, _e.n_cells])
            for e, _e in zip(ens, self._ens)
        ]

        tf.logging.info(g.shape)
        tf.logging.info(lstm_output.shape)
        tf.logging.info([e.shape for e in ens])
        # Return
        return (ens, g, lstm_output), final_state

    def get_all_variables(self):
        return (super(GridCellsRNN, self).get_variables() +
                self._core.get_variables())
