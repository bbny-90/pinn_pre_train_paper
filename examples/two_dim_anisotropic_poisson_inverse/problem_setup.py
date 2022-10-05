import os
import pathlib
from typing import List
import numpy as np
import pandas as pd
import sympy as sp

pjoin = os.path.join
SCRIPT_DIR = os.path.abspath(pathlib.Path(__file__).parent.absolute())
PROBLEM_SETUP_DIR = pjoin(SCRIPT_DIR, "data")
RANDOM_STATE = 0
X_LEFT, X_RIGHT = -1., 1.
Y_BOTTOM, Y_TOP = -1., 1.
NX_PDE, NY_PDE = 30, 30
NX_GUIDE, NY_GUIDE = 47, 47
N_SIDE_BC = 100
PERM = [[1., 1.],[1., 2.]]

PI = np.pi
PI_SP = sp.pi

class SolverSym():
    from sympy.core.expr import Expr
    def __init__(self, perm:List[List[float]]) -> None:
        from sympy.abc import x, y
        from helper.symbolic_calculator import (
            EvalNp, 
            get_gradient_scalar, 
            get_divergence_vector
        )
        u = sp.sin(sp.pi*y) * sp.sin(sp.pi*x)
        x_ = [x, y]
        grad_u = get_gradient_scalar(u, x_)
        flux = self._calc_flux(grad_u, perm)
        source = get_divergence_vector(flux, x_)
        
        self.eval_np_u = EvalNp(x_, u, 1)
        self.eval_np_source = EvalNp(x_, source, 1)
        self.eval_np_flux = [EvalNp(x_, f, 1) for f in flux]
    
    def _calc_flux(self, grad:List[Expr], perm:List[List[float]]) -> List[Expr]:
        flux = []
        for i in range(len(perm)):
            tmp = 0.
            for j in range(len(perm[0])):
                tmp -= perm[i][j] * grad[j]
            flux.append(tmp)
        return flux

    def get_source(self, x:np.ndarray):
        return self.eval_np_source(x).flatten()

    def get_solution(self, x:np.ndarray):
        return self.eval_np_u(x).flatten()
    
    def get_flux(self, x:np.ndarray) -> np.ndarray:
        out = []
        for f in self.eval_np_flux:
            out.append(f(x))
        
        out = np.stack(out, axis=1)
        return out


def get_indomain_collocation_points(
    x_left, x_right, nx,
    y_bottom, y_top, ny
    )->np.ndarray:
    x_col = np.linspace(x_left, x_right, nx, endpoint=False)[1:]
    y_col = np.linspace(y_bottom, y_top, ny, endpoint=False)[1:]
    x_col, y_col = np.meshgrid(x_col, y_col)
    x_col = x_col.flatten().reshape(-1, 1)
    y_col = y_col.flatten().reshape(-1, 1)
    xy_col = np.hstack((x_col, y_col))
    return xy_col

def get_bc_pnts(
    x_left, x_right,
    y_bottom, y_top,
    Nside
)->np.ndarray:
    x_bc, y_bc = [], []
    # bottom
    x_bc.append(np.linspace(x_left, x_right, Nside))
    y_bc.append(np.ones_like(x_bc[-1])*y_bottom)
    # top
    x_bc.append(np.linspace(x_left, x_right, Nside))
    y_bc.append(np.ones_like(x_bc[-1])*y_top)
    # left
    y_bc.append(np.linspace(y_bottom, y_top, Nside, endpoint=False)[1:])
    x_bc.append(np.ones_like(y_bc[-1])*x_left)
    # right
    y_bc.append(np.linspace(y_bottom, y_top, Nside, endpoint=False)[1:])
    x_bc.append(np.ones_like(y_bc[-1])*x_right)
    # gather
    x_bc = np.concatenate(x_bc)
    y_bc = np.concatenate(y_bc)
    xy_bc = np.hstack((x_bc.reshape(-1, 1), y_bc.reshape(-1, 1)))
    return xy_bc


def create_data():
    if not os.path.exists(PROBLEM_SETUP_DIR):
        os.makedirs(PROBLEM_SETUP_DIR)
    
    symb_sol = SolverSym(PERM)
    
    x_pde = get_indomain_collocation_points(
        X_LEFT, X_RIGHT, NX_PDE,
        Y_BOTTOM, Y_TOP, NY_PDE
    )
    x_guide = get_indomain_collocation_points(
        X_LEFT, X_RIGHT, NX_GUIDE,
        Y_BOTTOM, Y_TOP, NY_GUIDE
    )

    source_pde = symb_sol.get_source(x_pde).flatten()

    u_guide = symb_sol.get_solution(x_guide).flatten()
    flux_guide = symb_sol.get_flux(x_guide)
    
    x_bc = get_bc_pnts(
        X_LEFT, X_RIGHT,
        Y_BOTTOM, Y_TOP,
        N_SIDE_BC
    )
    u_bc = symb_sol.get_solution(x_bc).flatten()

    
    pd.DataFrame(
        {'x0':x_pde[:,0], 'x1':x_pde[:,1], 'source':source_pde,
        }
    ).to_csv(pjoin(PROBLEM_SETUP_DIR, "pde_data.csv"), index=False)
    pd.DataFrame(
        {'x0':x_guide[:, 0], 'x1':x_guide[:, 1], 'u':u_guide, 
        'flux0':flux_guide[:,0], 'flux1':flux_guide[:,1],
        }
    ).to_csv(pjoin(PROBLEM_SETUP_DIR, "exp_data.csv"), index=False)
    pd.DataFrame(
        {'x0':x_bc[:,0], 'x1':x_bc[:,1], 'u':u_bc,
        }
    ).to_csv(pjoin(PROBLEM_SETUP_DIR, "dbc_data.csv"), index=False)
    

if __name__ == "__main__":
    create_data()