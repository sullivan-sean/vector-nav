import torch
import ipykernel
from namedtensor import ntorch


def print_mem():
    print(f"Mem (MB): {torch.cuda.memory_allocated() / (1024 ** 2)}")
    print(f"Cached Mem (MB): {torch.cuda.memory_cached() / (1024 ** 2)}")
    print(f"Max Mem (MB): {torch.cuda.max_memory_allocated() / (1024 ** 2)}")


def in_ipynb():
    try:
        cfg = get_ipython()
        return isinstance(cfg, ipykernel.zmqshell.ZMQInteractiveShell)
    except NameError:
        return False


def unit_vector(angle):
    """Return tensor of unit vector in direction of angle"""
    return torch.stack((angle.cos(), angle.sin()))


def angle_between(u, v):
    """Find angle between two vectors"""
    # d = torch.sum(u * v, dim=1)
    d = torch.einsum("ij,ij->j", u, v)
    denom = torch.norm(u, p=2, dim=0) * torch.norm(v, p=2, dim=0)
    eps = 1 - 1e-7
    return torch.acos(torch.clamp(d / denom, min=-eps, max=eps))


def rotation_matrix(theta):
    """Find 2d rotation matrix for a given angle"""
    cos, sin = unit_vector(theta)
    top = torch.stack((cos, -sin))
    bot = torch.stack((sin, cos))
    return torch.stack((top, bot))


DIMS = [
    ('batch', 't', 'ax'),
    ('batch', 't', 'hd'),
    ('batch', 't', 'input'),
    ('batch', 'ax'),
    ('batch', 'hd'),
]


def get_batch(traj, place_cells, hd_cells, dims=None, pos=False):
    if dims is None:
        dims = DIMS
    ntraj = [ntorch.tensor(i, names=n).cuda() for i, n in zip(traj, dims)]
    target_pos, target_hd, ego_vel, init_pos, init_hd = ntraj
    cs, c0 = place_cells(target_pos), place_cells(init_pos)
    hs, h0 = hd_cells(target_hd), hd_cells(init_hd)

    hs = hs[{'hd': 0}]
    h0 = h0[{'hd': 0}]

    if pos:
        return cs, hs, ego_vel, c0, h0, target_pos

    return cs, hs, ego_vel, c0, h0
