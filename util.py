import torch


def print_mem():
    print(f"Mem (MB): {torch.cuda.memory_allocated() / (1024 ** 2)}")
    print(f"Cached Mem (MB): {torch.cuda.memory_cached() / (1024 ** 2)}")
    print(f"Max Mem (MB): {torch.cuda.max_memory_allocated() / (1024 ** 2)}")


def in_ipynb():
    try:
        cfg = get_ipython().config
        if cfg["IPKernelApp"]["parent_appname"] == "ipython-notebook":
            return True
        else:
            return False
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
