from model import CrossEntropyLoss
import numpy as np
from namedtensor import ntorch
import torch
from viz import visualize_g
from util import in_ipynb

if in_ipynb():
    from tqdm import tqdm_notebook as tqdm
else:
    from tqdm import tqdm


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


def clip_layers(layers, clip):
    """clip gradients in layers to range before weight updates"""
    for l in layers:
        for p in l.parameters():
            p.grad.data.clamp_(-clip, clip)


def train_model(
    model,
    dataloader,
    num_epochs=10,
    lr=1e-5,
    momentum=0.9,
    weight_decay=1e-5,
    clip=1e-5,
):
    """Train model using CrossEntropy and RMSProp as in paper"""

    hdloss = CrossEntropyLoss().spec("hdcell")
    placeloss = CrossEntropyLoss().spec("placecell")

    params = decay_params(model, ["head", "place"], weight_decay)
    optimizer = torch.optim.RMSprop(params, lr=lr, momentum=0.9)

    losses = []

    for k in range(num_epochs):
        model.train()
        epoch_losses = []

        for i, (inp, cs, hs, _) in enumerate(tqdm(dataloader)):
            inp, cs, hs = [
                ntorch.tensor(i, names=("batch", "t", d)).cuda()
                for i, d in zip(
                    [inp, cs, hs], ["input", "placecell", "hdcell"]
                )
            ]

            optimizer.zero_grad()

            zs, ys, _ = model(inp, cs[{"t": 0}], hs[{"t": 0}])

            loss = hdloss(zs, hs) + placeloss(ys, cs)
            epoch_losses.append(loss.item())

            loss.backward()
            clip_layers([model.head, model.place], clip)
            optimizer.step()

            del inp, cs, hs
            torch.cuda.empty_cache()

        losses += epoch_losses

        # Output and visualize progress each epoch
        print(
            f"epoch {k}, mean loss {np.mean(epoch_losses)}, std loss {np.std(epoch_losses)}"
        )
        visualize_g(model, dataloader)

    return losses
