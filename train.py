from model import CrossEntropyLoss
import numpy as np
from namedtensor import ntorch
import torch
from viz import visualize_g
from util import in_ipynb, get_batch
from tqdm import tqdm_notebook, tqdm


def decay_params(model, layers, decay):
    """Apply decay to layers in model, return params for optimizer"""
    layers = [getattr(model, layer) for layer in layers]
    params = [{"params": set(model.parameters())}]

    for layer in layers:
        layer_params = layer.parameters()
        params[0]["params"] -= set(layer_params)
        params.append({"params": layer_params, "weight_decay": decay})

    params[0]["params"] = list(params[0]["params"])
    return params


def train_model(
    model,
    dataloader,
    place_cells,
    hd_cells,
    num_epochs=10,
    lr=1e-5,
    momentum=0.9,
    weight_decay=1e-5,
    clip=1e-5,
):
    """Train model using CrossEntropy and RMSProp as in paper"""

    hdloss = CrossEntropyLoss().spec("hdcell")
    placeloss = CrossEntropyLoss().spec("placecell")

    params = decay_params(model, ["head", "place", "g"], weight_decay)
    optimizer = torch.optim.RMSprop(params, lr=lr, momentum=momentum)

    losses = []

    tq = tqdm_notebook if in_ipynb() else tqdm

    for k in range(num_epochs):
        model.train()
        epoch_losses = []

        for i, traj in enumerate(tq(dataloader)):
            cs, hs, ego_vel, c0, h0 = get_batch(traj, place_cells, hd_cells)

            optimizer.zero_grad()

            zs, ys, _ = model(ego_vel, c0, h0)

            loss = hdloss(zs, hs) + placeloss(ys, cs)
            epoch_losses.append(loss.item())

            loss.backward()

            # torch.nn.utils.clip_grad_value_(model.head.parameters(), clip)
            # torch.nn.utils.clip_grad_value_(model.place.parameters(), clip)
            torch.nn.utils.clip_grad_value_(model.parameters(), clip)

            optimizer.step()

            if (i + 1) % 1000 == 0 or i + 1 == len(dataloader):
                # Output and visualize progress each epoch
                print(
                    f"epoch {k}, mean loss {np.mean(epoch_losses)}, std loss {np.std(epoch_losses)}"
                )
                visualize_g(model, dataloader, place_cells, hd_cells)
                model.train()
                break
            if i > 1000 * num_epochs:
                return epoch_losses
        losses += epoch_losses
    return losses
