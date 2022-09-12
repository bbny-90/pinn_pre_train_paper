import inspect

def test_laplace_1d():
    from pdes.operators import get_laplace_scalar
    from helper.other import get_torch_device
    import torch
    import numpy as np
    device = get_torch_device()
    ndata = 10
    x = torch.rand(ndata, 1).float().to(device=device)
    x.requires_grad_(True)
    y = x**3
    lap_true = (6. * x).detach().numpy()
    lap_autodiff = get_laplace_scalar(y, x, device=device).detach().numpy()
    assert np.allclose(lap_autodiff, lap_true)
    print(f"{inspect.stack()[0][3]} is passed")

def test_laplace_3d():
    from pdes.operators import get_laplace_scalar
    from helper.other import get_torch_device
    import torch
    import numpy as np
    device = get_torch_device()
    ndata, ndim = 10, 3
    x = torch.rand(ndata, ndim).float().to(device=device)
    x.requires_grad_(True)
    y = x.pow(3).sum(dim=1).view(-1, 1)
    lap_true = 6. * x.sum(dim=1).detach().numpy().reshape(-1, 1)
    lap_autodiff = get_laplace_scalar(y, x, device=device).detach().numpy()
    assert np.allclose(lap_autodiff, lap_true)
    print(f"{inspect.stack()[0][3]} is passed")

if __name__ == "__main__":
    test_laplace_1d()
    test_laplace_3d()