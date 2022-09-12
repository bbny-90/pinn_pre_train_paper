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


def get_laplace_scalar(
    y:torch.tensor, x:torch.tensor, device:torch.device = None
    ) -> torch.tensor:
    """
        this function calculates the laplace operator for a scalar-valued
        function in m-dim space
        x: independed variable (n, m)
        y: dependent variable (n, 1)
    """
    assert y.dim() == x.dim() == 2
    assert y.shape[1] == 1
    assert y.shape[0] == x.shape[0]
    space_dim = x.shape[1]
    num_samples = y.size()[0]    
    dydx = grad(y, x, torch.ones(num_samples, 1, device=device),
                        create_graph=True, retain_graph=True)[0]
    tmp = torch.zeros_like(y)
    for i in range(space_dim):
        tmp += grad(dydx[:,i:i+1], x, torch.ones(num_samples, 1, device=device),
                            create_graph=True, retain_graph=True)[0][:, i:i+1]
    return tmp