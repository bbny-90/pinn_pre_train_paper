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
    # tmp = torch.zeros(num_samples, 1)
    # for i in range(space_dim):
    #     tmp += grad(y[:,i:i+1], x, torch.ones(num_samples, 1, device=device),
    #                         create_graph=True, retain_graph=True)[0][:, i:i+1]
    tmp = []
    for i in range(space_dim):
        tmp.append(
            grad(y[:,i:i+1], x, torch.ones(num_samples, 1, device=device),
                            create_graph=True, retain_graph=True)[0][:, i:i+1]
        )
    return torch.concat(tmp, dim=1).sum(dim=1).view(-1, 1)

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


def get_gradient_vector(
    y:torch.tensor, x:torch.tensor, device:torch.device = None
    ) -> torch.tensor:
    """
        this function calculates the gradient of a vector_valued
        function in m-dim space
        x: independed variable (n, mx)
        y: dependent variable (n, my)
    """
    ndata, ndimY = y.shape
    ndimX = x.shape[1]
    out = torch.zeros(ndata, ndimY, ndimX).float().to(device=device)
    for i in range(ndimY):
        for j in range(ndimX):
            out[:, i, j] += get_grad_scalar(y[:, i:i+1], x, device)[:, j]
    return out

def get_divergence_tensor(
    y:torch.tensor, x:torch.tensor, device:torch.device = None
    ) -> torch.tensor:
    """
        this function calculates the gradient of a vector_valued
        function in m-dim space
        x: independed variable (n, m)
        y: dependent variable (n, m, m)
    """
    out = []
    for k in range(y.shape[-1]):
        out.append(get_divergence_vector(y[:, :, k], x, device))
    out = torch.concat(out, dim=1)
    return out