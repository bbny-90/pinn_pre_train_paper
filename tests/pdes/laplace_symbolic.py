import inspect

def test_laplace_1d():
    import numpy as np
    from sympy.abc import x
    from helper.symbolic_calculator import get_laplacian, EvalNp
    
    ndata, ndim = 10, 1
    x_np = np.random.rand(ndata, ndim)
    lap_u_true = x_np[:,0] * 1.

    u = x**3 / 6.
    lap_u = get_laplacian(u, [x])
    eval_np_lap_u = EvalNp([x], lap_u, 1)
    lap_u_symb_np = eval_np_lap_u(x_np)
    assert np.allclose(lap_u_true, lap_u_symb_np), (lap_u_true, lap_u_symb_np)
    print(f"{inspect.stack()[0][3]} is passed")

def test_laplace_3d():
    import numpy as np
    from sympy.abc import x, y
    from helper.symbolic_calculator import get_laplacian, EvalNp
    
    ndata, ndim = 10, 2
    x_np = np.random.rand(ndata, ndim)
    u_np = 0.5*x_np[:, 0]**2*x_np[:, 1] + 0.5*x_np[:, 1]**2*x_np[:, 0]
    lap_u_true = np.sum(x_np, axis=1)

    u = 0.5 * x**2 * y + 0.5 * y**2 * x
    lap_u = get_laplacian(u, [x, y])
    eval_np_lap_u = EvalNp([x, y], lap_u, 1)
    lap_u_symb_np = eval_np_lap_u(x_np)
    assert np.allclose(lap_u_true, lap_u_symb_np)
    print(f"{inspect.stack()[0][3]} is passed")

if __name__ == "__main__":
    test_laplace_1d()
    test_laplace_3d()