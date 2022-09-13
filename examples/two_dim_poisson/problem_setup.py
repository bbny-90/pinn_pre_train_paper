import os
import pathlib
import numpy as np
import pandas as pd
import sympy as sp
from pyDOE import lhs

pjoin = os.path.join
SCRIPT_DIR = os.path.abspath(pathlib.Path(__file__).parent.absolute())
PROBLEM_SETUP_DIR = pjoin(SCRIPT_DIR, "data")
RANDOM_STATE = 0
X_LEFT, X_RIGHT = 0., 1.
Y_BOTTOM, Y_TOP = 0., 1.
NX, NY = 40, 40
NUM_VAL_PNTS = 400
PI = np.pi
PI_SP = sp.pi

class SymbSol():
    def __init__(self) -> None:
        from sympy.abc import x, y
        u = sp.sin(4.*PI_SP*y) * sp.exp(-4.*PI_SP*x)
        self.x = [x, y]
        self.u = u

def get_indomain_collocation_points()->np.ndarray:
    x = np.linspace(X_LEFT, X_RIGHT, NX, endpoint=False)[1:]
    y = np.linspace(Y_BOTTOM, Y_TOP, NY, endpoint=False)[1:]
    x, y = np.meshgrid(x, y)
    x = x.flatten().reshape(-1, 1)
    y = y.flatten().reshape(-1, 1)
    xy = np.hstack((x, y))
    return xy

def get_bc_pnts()->np.ndarray:
    x_bc, y_bc = [], []
    # bottom
    x_bc.append(np.linspace(X_LEFT, X_RIGHT, NX))
    y_bc.append(np.ones_like(x_bc[-1])*Y_BOTTOM)
    # top
    x_bc.append(np.linspace(X_LEFT, X_RIGHT, NX))
    y_bc.append(np.ones_like(x_bc[-1])*Y_TOP)
    # left
    y_bc.append(np.linspace(Y_BOTTOM, Y_TOP, NY, endpoint=False)[1:])
    x_bc.append(np.ones_like(y_bc[-1])*X_LEFT)
    # right
    y_bc.append(np.linspace(Y_BOTTOM, Y_TOP, NY, endpoint=False)[1:])
    x_bc.append(np.ones_like(y_bc[-1])*X_RIGHT)
    # all
    x_bc = np.concatenate(x_bc)
    y_bc = np.concatenate(y_bc)
    xy_bc = np.hstack((x_bc.reshape(-1, 1), y_bc.reshape(-1, 1)))
    return xy_bc

def get_exact_solution(x:np.ndarray)->np.ndarray:
    w = 4.*PI
    return np.sin(w*x[:, 1])*np.exp(-w*x[:, 0])



def create_data():
    if not os.path.exists(PROBLEM_SETUP_DIR):
        os.makedirs(PROBLEM_SETUP_DIR)
    from helper.symbolic_calculator import EvalNp, get_laplacian   
    symb_sol = SymbSol()
    lap_u = get_laplacian(symb_sol.u, symb_sol.x)
    eval_np_u = EvalNp(symb_sol.x, symb_sol.u, 1)
    eval_np_lap_u = EvalNp(symb_sol.x, lap_u, 1)
    
    x_pde = get_indomain_collocation_points()
    u_pde = eval_np_u(x_pde)
    source_pde = eval_np_lap_u(x_pde).flatten()
    
    x_bc = get_bc_pnts()
    u_bc = eval_np_u(x_bc)
    
    np.random.seed(RANDOM_STATE)
    x_vald = lhs(2, NUM_VAL_PNTS)
    x_vald[:, 0] = x_vald[:, 0] * (X_RIGHT - X_LEFT) + X_LEFT
    x_vald[:, 1] = x_vald[:, 1] * (Y_TOP - Y_BOTTOM) + Y_BOTTOM
    np.random.seed(None)
    u_vald = eval_np_u(x_vald)
    source_vald = eval_np_lap_u(x_vald).flatten()

    
    pd.DataFrame({'x0':x_pde[:,0], 'x1':x_pde[:,1], 'u':u_pde, 'source':source_pde}
    ).to_csv(pjoin(PROBLEM_SETUP_DIR, "pde_data.csv"), index=False)
    pd.DataFrame({'x0':x_vald[:, 0], 'x1':x_vald[:, 1], 'u':u_vald, 'source':source_vald}
    ).to_csv(pjoin(PROBLEM_SETUP_DIR, "validation_data.csv"), index=False)
    pd.DataFrame({'x0':x_bc[:, 0], 'x1':x_bc[:, 1], 'u':u_bc}
    ).to_csv(pjoin(PROBLEM_SETUP_DIR, "bc_data.csv"), index=False)
    

if __name__ == "__main__":
    create_data()