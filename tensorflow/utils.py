import os
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

np.seterr(invalid="ignore")


def clip_all_gradients(g, var, limit):
    return (tf.clip_by_value(g, -limit, limit), var)


def concat_dict(acc, new_data):
    """Dictionary concatenation function."""

    def to_array(kk):
        if isinstance(kk, np.ndarray):
            return kk
        else:
            return np.asarray([kk])

    for k, v in new_data.items():
        if isinstance(v, dict):
            if k in acc:
                acc[k] = concat_dict(acc[k], v)
            else:
                acc[k] = concat_dict(dict(), v)
        else:
            v = to_array(v)
            if k in acc:
                acc[k] = np.concatenate([acc[k], v])
            else:
                acc[k] = np.copy(v)
    return acc


def get_scores_and_plot(
        scorer,
        data_abs_xy,
        activations,
        directory,
        filename,
        plot_graphs=True,  # pylint: disable=unused-argument
        nbins=20,  # pylint: disable=unused-argument
        cm="jet",
        sort_by_score_60=True):
    """Plotting function."""

    # Concatenate all trajectories
    xy = data_abs_xy.reshape(-1, data_abs_xy.shape[-1])
    act = activations.reshape(-1, activations.shape[-1])
    n_units = act.shape[1]
    # Get the rate-map for each unit
    s = [
        scorer.calculate_ratemap(xy[:, 0], xy[:, 1], act[:, i])
        for i in range(n_units)
    ]
    # Get the scores
    score_60, score_90, max_60_mask, max_90_mask, sac = zip(
        *[scorer.get_scores(rate_map) for rate_map in s])
    # Separations
    # separations = map(np.mean, max_60_mask)
    # Sort by score if desired
    if sort_by_score_60:
        ordering = np.argsort(-np.array(score_60))
    else:
        ordering = range(n_units)
    # Plot
    cols = 16
    rows = int(np.ceil(n_units / cols))
    fig = plt.figure(figsize=(24, rows * 4))
    for i in range(n_units):
        rf = plt.subplot(rows * 2, cols, i + 1)
        acr = plt.subplot(rows * 2, cols, n_units + i + 1)
        if i < n_units:
            index = ordering[i]
            title = "%d (%.2f)" % (index, score_60[index])
            # Plot the activation maps
            scorer.plot_ratemap(s[index], ax=rf, title=title, cmap=cm)
            # Plot the autocorrelation of the activation maps
            scorer.plot_sac(sac[index],
                            mask_params=max_60_mask[index],
                            ax=acr,
                            title=title,
                            cmap=cm)
    # Save
    if not os.path.exists(directory):
        os.makedirs(directory)
    with PdfPages(os.path.join(directory, filename), "w") as f:
        plt.savefig(f, format="pdf")
    plt.close(fig)
    return (np.asarray(score_60), np.asarray(score_90),
            np.asarray(map(
                np.mean, max_60_mask)), np.asarray(map(np.mean, max_90_mask)))
