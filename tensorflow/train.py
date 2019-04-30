import matplotlib
import numpy as np
import tensorflow as tf
import tkinter  # pylint: disable=unused-import
from h5file import Saver
import tftables
import contextlib


matplotlib.use('Agg')

import dataset_reader  # pylint: disable=g-bad-import-order, g-import-not-at-top
import model  # pylint: disable=g-bad-import-order
import scores  # pylint: disable=g-bad-import-order
import utils  # pylint: disable=g-bad-import-order
import ensembles

# Task config
tf.flags.DEFINE_string('task_root', None, 'Dataset path.')
tf.flags.DEFINE_integer('task_neurons_seed', 8341, 'Seeds.')
tf.flags.DEFINE_string('saver_results_directory', None,
                       'Path to directory for saving results.')

# Require flags
tf.flags.mark_flag_as_required('task_root')
tf.flags.mark_flag_as_required('saver_results_directory')
FLAGS = tf.flags.FLAGS


def train(hidden_size=128, g_size=256, batch_size=10, seed=8341):
    tf.reset_default_graph()
    dataset = 'square_room'

    # Create the motion models for training and evaluation
    data_reader = dataset_reader.DataReader(dataset, root=FLAGS.task_root, num_threads=4)
    train_traj = data_reader.read(batch_size=batch_size)

    # Create the ensembles that provide targets during training
    pcs = ensembles.PlaceCellEnsemble(256, stdev=0.01, env_size=2.2, seed=seed)
    hds = ensembles.HeadDirectionCellEnsemble(12, 20., seed)

    # Model creation
    rnn = model.GridCellsRNN([pcs, hds], hidden_size, g_size, g_has_bias=False)

    init_pos, init_hd, ego_vel, target_pos, target_hd = train_traj
    inputs = tf.concat([ego_vel], axis=2)

    pc_0 = tf.squeeze(pcs.get_init(init_pos[:, tf.newaxis, :]), axis=1)
    hd_0 = tf.squeeze(hds.get_init(init_hd[:, tf.newaxis, :]), axis=1)

    outputs, _ = rnn([pc_0, hd_0], inputs, training=True)
    (pc_logits, hd_logits), bottleneck, lstm_output = outputs

    pc_loss = pcs.loss(pc_logits, target_pos, name='pc_loss')
    hd_loss = hds.loss(hd_logits, target_hd, name='hd_loss')
    train_loss = tf.reduce_mean(pc_loss + hd_loss, name='train_loss')

    optimizer = tf.train.RMSPropOptimizer(learning_rate=1e-5, momentum=0.9)
    grad = optimizer.compute_gradients(train_loss)
    clipped_grad = [utils.clip_all_gradients(g, var, 1e-5) for g, var in grad]
    train_op = optimizer.apply_gradients(clipped_grad)

    # Store the grid scores
    grid_scores = dict(
        btln_60=np.zeros((g_size,)),
        btln_90=np.zeros((g_size,)),
        btln_60_separation=np.zeros((g_size,)),
        btln_90_separation=np.zeros((g_size,)),
        lstm_60=np.zeros((hidden_size,)),
        lstm_90=np.zeros((hidden_size,))
    )

    # Create scorer objects
    starts = [0.2] * 10
    ends = np.linspace(0.4, 1.0, num=10)
    masks_parameters = zip(starts, ends.tolist())
    latest_epoch_scorer = scores.GridScorer(20, data_reader.get_coord_range(),
                                            masks_parameters)
    if False:
        S = Saver('testfile.hdf5')
        with tf.train.SingularMonitoredSession() as sess:
            for j in range(1000000):
                if (j + 1) % 100 == 0:
                    print(j + 1)
                S.save_traj(i.eval(session=sess) for i in train_traj)

    with tf.train.SingularMonitoredSession() as sess:
        for epoch in range(1000):
            loss_acc = list()
            for _ in range(1000):
                res = sess.run({
                    'train_op': train_op,
                    'total_loss': train_loss
                })
                loss_acc.append(res['total_loss'])

            tf.logging.info('Epoch %i, mean loss %.5f, std loss %.5f', epoch,
                            np.mean(loss_acc), np.std(loss_acc))
            if epoch % 2 == 0:
                res = dict()
                for _ in range(4000 // batch_size):
                    mb_res = sess.run({
                        'bottleneck': bottleneck,
                        'lstm': lstm_output,
                        'pos_xy': target_pos
                    })
                    res = utils.concat_dict(res, mb_res)

                # Store at the end of validation
                filename = 'rates_and_sac_latest_hd.pdf'
                grid_scores['btln_60'], grid_scores['btln_90'], grid_scores[
                    'btln_60_separation'], grid_scores[
                        'btln_90_separation'] = utils.get_scores_and_plot(
                            latest_epoch_scorer, res['pos_xy'],
                            res['bottleneck'], FLAGS.saver_results_directory,
                            filename)


def main(unused_argv):
    tf.logging.set_verbosity(3)  # Print INFO log messages.
    train()


if __name__ == '__main__':
    tf.app.run()
