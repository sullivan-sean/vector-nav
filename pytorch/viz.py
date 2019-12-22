from namedtensor import ntorch
import matplotlib.pyplot as plt
import torch
from scipy import stats
from util import get_batch


def visualize_g(model, test_iter, place_cells, hd_cells, offset=0, limit=50):
    """Visualize 25 cells in G layer of model (applied to output of LSTM)"""
    model.eval()
    G, P = None, None
    c = 0

    # Get batches up to limit as samples
    for traj in test_iter:
        cs, hs, ego_vel, c0, h0, xs = get_batch(traj, place_cells, hd_cells, pos=True)
        if c > limit:
            break
        zs, gs, ys = model(ego_vel, c0, h0)
        if G is None:
            G = gs.cpu()
            P = xs.cpu()
        else:
            G = ntorch.cat((G, gs.cpu()), "batch")
            P = ntorch.cat((P, xs.cpu()), "batch")
        del ego_vel, cs, xs, hs, zs, ys, gs, h0, c0
        torch.cuda.empty_cache()
        c += 1

    pts = P.stack(("t", "batch"), "pts")
    G = G.stack(("t", "batch"), "pts")

    xs, ys = [pts.get("ax", i).values.detach().numpy() for i in [0, 1]]

    # Plot 5x5 grid of cell activations, starting at offset
    axs = plt.subplots(5, 5, figsize=(50, 50))[1]
    axs = axs.flatten()

    for i, ax in enumerate(axs):
        acts = G.get("placecell", offset + i).values.detach().numpy()
        res = stats.binned_statistic_2d(
            xs, ys, acts, bins=20, statistic="mean"
        )[0]
        ax.imshow(res, cmap="jet")
        ax.axis("off")
    plt.show()

