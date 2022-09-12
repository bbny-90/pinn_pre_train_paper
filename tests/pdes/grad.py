import inspect

def test_first_grad_scalar():
    from pdes.operators import get_laplace_scalar
    from pdes.operators import get_grad_scalar
    from helper.other import get_torch_device
    import torch
    import numpy as np
    device = get_torch_device()
    ndata = 10
    pi = 3.14
    x = torch.rand(ndata, 1).float().to(device=device)
    x.requires_grad_(True)
    y = x**2 + torch.sin(pi * x)
    dydx_true = 2. * x + pi * torch.cos(pi * x)
    dydx_true = dydx_true.detach().numpy()
    dydx_autodiff = get_grad_scalar(y, x, device=device).detach().numpy()
    assert np.allclose(dydx_true, dydx_autodiff)
    print(f"{inspect.stack()[0][3]} is passed")

def test_second_grad_scalar():
    from pdes.operators import get_grad_scalar
    from helper.other import get_torch_device
    import torch
    import numpy as np
    device = get_torch_device()
    ndata = 10
    pi = 3.14
    x = torch.rand(ndata, 1).float().to(device=device)
    x.requires_grad_(True)
    y = x**2 + torch.sin(pi * x)
    dyydxx_true = 2. - pi**2 * torch.sin(pi * x)
    dyydxx_true = dyydxx_true.detach().numpy()
    #
    dydx_autodiff = get_grad_scalar(y, x, device=device)
    ddydxx_autodiff = get_grad_scalar(dydx_autodiff, x, device=device).detach().numpy()
    assert np.allclose(ddydxx_autodiff, dyydxx_true)
    print(f"{inspect.stack()[0][3]} is passed")

if __name__ == "__main__":
    test_first_grad_scalar()
    test_second_grad_scalar()