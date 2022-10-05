from typing import List, Callable
import numpy as np
import sympy as sp 
from sympy.core.symbol import Symbol
from sympy.core.expr import Expr

class EvalNp():
    def __init__(self, x:List[Symbol], u:Expr, u_dim) -> None:
        self.u_dim = u_dim
        self.eval_np:Callable[[np.ndarray]] = sp.lambdify(x, u, "numpy")
    
    def __call__(self, x:np.ndarray) -> np.ndarray:
        ndim = x.shape[1]
        args = tuple([x[:, i] for i in range(ndim)])
        out = self.eval_np(*args)
        ndata = x.shape[0]
        if isinstance(out, int) or isinstance(out, float):
            out = out * np.ones((ndata, self.u_dim))
        return out

def get_gradient_scalar(u:Expr, x:List[Symbol])->List[Expr]:
    grad = []
    for xi in x:
        grad.append(sp.diff(u, xi))
    return grad

def get_divergence_vector(u:List[Expr], x:List[Symbol])->Expr:
    assert len(u) == len(x)
    div = 0.
    for ui, xi in zip(u, x):
        div += sp.diff(ui, xi)
    return div

def get_laplacian(u:Expr, x:List[Symbol])->Expr:
    grad = get_gradient_scalar(u, x)
    return get_divergence_vector(grad)