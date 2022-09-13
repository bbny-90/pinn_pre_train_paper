from typing import List, Callable
import numpy as np
import sympy as sp 
# from sympy.abc import x, y
from sympy.core.symbol import Symbol
from sympy.core.expr import Expr

class EvalNp():
    def __init__(self, x:List[Symbol], u:Expr) -> None:
        self.eval_np:Callable[[np.ndarray]] = sp.lambdify(x, u, "numpy")
    
    def __call__(self, x:np.ndarray) -> np.ndarray:
        ndim = x.shape[1]
        args = tuple([x[:, i] for i in range(ndim)])
        out = self.eval_np(*args)
        if isinstance(out, int) or isinstance(out, float):
            out = out * np.ones_like(x)
        return out

def get_laplacian(u:Expr, x:List[Symbol]):
    lap = 0.
    for xi in x:
        lap += sp.diff(sp.diff(u, xi), xi)
    return lap