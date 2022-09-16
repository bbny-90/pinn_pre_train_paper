import inspect

def test_first_grad_scalar():
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

def test_grad_scalar_2dim():
    from pdes.operators import get_grad_scalar
    from helper.other import get_torch_device
    import torch
    import numpy as np
    device = get_torch_device()
    ndata, ndim = 10, 2
    x = torch.rand(ndata, ndim).float().to(device=device)
    x.requires_grad_(True)
    y = x[:,0:1]**2 + x[:,0:1]*x[:,1:2] + x[:,1:2]**2
    dydx_true = torch.zeros_like(x)
    dydx_true[:, 0] = 2.*x[:,0] + x[:,1]
    dydx_true[:, 1] = 2.*x[:,1] + x[:,0]
    dydx_true = dydx_true.detach().numpy()
    dydx_autodiff = get_grad_scalar(y, x, device=device).detach().numpy()
    assert np.allclose(dydx_true, dydx_autodiff)
    print(f"{inspect.stack()[0][3]} is passed")

def test_grad_vector_2dim():
    from pdes.operators import get_gradient_vector
    from helper.other import get_torch_device
    import torch
    import numpy as np
    
    device = get_torch_device()
    ndata, ndim = 10, 2
    x = torch.rand(ndata, ndim).float().to(device=device)
    x.requires_grad_(True)
    y0 = x[:, 1] * x[:, 0]
    y1 = x[:, 0]**2 + x[:, 1]
    y = torch.stack([y0, y1], dim=1)
    grad_y = get_gradient_vector(y, x, device).detach().numpy()
    grad_y_true = np.zeros((ndata, ndim, ndim))
    x = x.detach().numpy()
    grad_y_true[:, 0, 0] = x[:, 1]
    grad_y_true[:, 0, 1] = x[:, 0]
    grad_y_true[:, 1, 0] = 2. * x[:, 0]
    grad_y_true[:, 1, 1] = 1.
    assert np.allclose(grad_y, grad_y_true)
    print(f"{inspect.stack()[0][3]} is passed")

def test_divergence_tensor_2dim():
    from pdes.operators import get_gradient_vector, get_divergence_tensor
    from helper.other import get_torch_device
    import torch
    import numpy as np
    
    device = get_torch_device()
    ndata, ndim = 4, 2
    x = torch.rand(ndata, ndim).float().to(device=device)
    x.requires_grad_(True)
    y0 = x[:, 1]**2 * x[:, 0]
    y1 = x[:, 0]**3 + x[:, 1]
    y = torch.stack([y0, y1], dim=1)
    grad_y = get_gradient_vector(y, x, device)
    div_grad_y = get_divergence_tensor(grad_y, x, device=device).detach().numpy()
    x = x.detach().numpy()
    
    grad_y_true = np.zeros((ndata, ndim, ndim))
    grad_y_true[:, 0, 0] = x[:, 1]**2
    grad_y_true[:, 0, 1] = x[:, 0] * x[:, 1] * 2
    grad_y_true[:, 1, 0] = 3. * x[:, 0]**2
    grad_y_true[:, 1, 1] = 1.

    div_grad_y_true = np.zeros((ndata, ndim))
    div_grad_y_true[:, 1] = x[:, 1] * 2

    assert np.allclose(div_grad_y, div_grad_y_true)
    print(f"{inspect.stack()[0][3]} is passed")

def test_biharmonic_2dim():
    from pdes.operators import get_gradient_vector, get_divergence_tensor
    from helper.other import get_torch_device
    import torch
    import numpy as np
    
    device = get_torch_device()
    ndata, ndim = 4, 2
    x = torch.rand(ndata, ndim).float().to(device=device)
    x.requires_grad_(True)
    u0 = x[:, 0]**3 + x[:, 1]**3
    u1 = x[:, 0] * x[:, 1]**2
    u = torch.stack([u0, u1], dim=1)
    grad_u = get_gradient_vector(u, x, device)
    eps = (grad_u + grad_u.transpose(1, 2)) * 0.5
    div_eps = get_divergence_tensor(eps, x, device=device).detach().numpy()
    eps = eps.detach().numpy()
    x = x.detach().numpy()
    
    eps_true = np.zeros((ndata, ndim, ndim))
    eps_true[:, 0, 0] = 3. * x[:, 0]**2
    eps_true[:, 1, 1] = x[:, 0] * x[:, 1] * 2
    eps_true[:, 0, 1] = 0.5 * (x[:, 1]**2 * 3 + x[:, 1]**2)
    eps_true[:, 1, 0] = eps_true[:, 0, 1]
    assert np.allclose(eps_true, eps)

    div_eps_true = np.zeros((ndata, ndim))
    div_eps_true[:, 0] = 6. * x[:, 0] + 0.5 * (x[:, 1]* 6. + x[:, 1]* 2.)
    div_eps_true[:, 1] = 2. * x[:, 0]
    assert np.allclose(div_eps_true, div_eps)

    print(f"{inspect.stack()[0][3]} is passed")

if __name__ == "__main__":
    test_first_grad_scalar()
    test_second_grad_scalar()
    test_grad_scalar_2dim()
    test_grad_vector_2dim()
    test_divergence_tensor_2dim()
    test_biharmonic_2dim()