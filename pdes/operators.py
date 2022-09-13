import torch
from torch.autograd import grad

def get_grad_scalar(
    y:torch.tensor, x:torch.tensor, device:torch.device = None
    ) -> torch.tensor:
    """
        this function calculates the gradient of a scalar-valued
        function in m-dim space
        x: independed variable (n, m)
        y: dependent scalar variable (n, 1)
    """
    assert y.dim() == x.dim() == 2
    assert y.shape[1] == 1
    assert y.shape[0] == x.shape[0]
    num_samples = y.size()[0]
    dy = grad(y, x, torch.ones(num_samples, 1, device=device),
                        create_graph=True, retain_graph=True)[0]
    assert dy.shape == x.shape
    return dy

def get_divergence_vector(
    y:torch.tensor, x:torch.tensor, device:torch.device = None
):
    """
        this function calculates the divergence of a vector-valued
        function in m-dim space
        x: independed variable (n, m)
        y: dependent variable (n, m)
    """
    assert y.dim() == x.dim() == 2
    assert y.shape[0] == x.shape[0]
    num_samples, space_dim = y.shape
    tmp = torch.zeros_like(y)
    for i in range(space_dim):
        tmp += grad(y[:,i:i+1], x, torch.ones(num_samples, 1, device=device),
                            create_graph=True, retain_graph=True)[0][:, i:i+1]
    return tmp

def get_laplace_scalar(
    y:torch.tensor, x:torch.tensor, device:torch.device = None
    ) -> torch.tensor:
    """
        this function calculates the laplace operator for a scalar-valued
        function in m-dim space
        x: independed variable (n, m)
        y: dependent variable (n, 1)
    """
    dydx = get_grad_scalar(y, x, device)
    return get_divergence_vector(dydx, x, device)