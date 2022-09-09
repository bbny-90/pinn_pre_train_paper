import torch
from torch.autograd import grad

def get_grad_scalar(
    y:torch.tensor, x:torch.tensor, device:torch.device = None
    ) -> torch.tensor:
    assert y.shape == x.shape, (y.shape == x.shape)
    assert y.dim() == 2
    assert y.shape[1] == 1
    dy = grad(y, x, torch.ones(x.size()[0], 1, device=device),
                        create_graph=True, retain_graph=True)[0]
    return dy