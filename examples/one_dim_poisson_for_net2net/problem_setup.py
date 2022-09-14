import os
import pathlib
import numpy as np
import pandas as pd
import sympy as sp

pjoin = os.path.join
SCRIPT_DIR = os.path.abspath(pathlib.Path(__file__).parent.absolute())
PROBLEM_SETUP_DIR = pjoin(SCRIPT_DIR, "data")
X_LEFT, X_RIGHT = -1., 1.
NUM_PDE_PNTS = 101
PI_SP = sp.pi

class SymbSol():
    def __init__(self) -> None:
        from sympy.abc import x
        u = sp.sin(1.2*PI_SP*x)
        self.x = [x]
        self.u = u

def create_data():
    if not os.path.exists(PROBLEM_SETUP_DIR):
        os.makedirs(PROBLEM_SETUP_DIR)
    from helper.symbolic_calculator import EvalNp, get_laplacian   
    symb_sol = SymbSol()
    lap_u = get_laplacian(symb_sol.u, symb_sol.x)
    eval_np_u = EvalNp(symb_sol.x, symb_sol.u, 1)
    eval_np_lap_u = EvalNp(symb_sol.x, lap_u, 1)

    x_pde = np.linspace(X_LEFT, X_RIGHT, NUM_PDE_PNTS, endpoint=False)[1:].reshape(-1, 1)
    u_pde = eval_np_u(x_pde).flatten()
    source_pde = eval_np_lap_u(x_pde).flatten()    
    
    x_bc = np.array([X_LEFT, X_RIGHT]).reshape(-1, 1)
    u_bc = eval_np_u(x_bc).flatten()
    
    pd.DataFrame({'x':x_pde.flatten(), 'u':u_pde, 'source':source_pde}
    ).to_csv(pjoin(PROBLEM_SETUP_DIR, "pde_data.csv"), index=False)
    pd.DataFrame({'x':x_bc.flatten(), 'u':u_bc}
    ).to_csv(pjoin(PROBLEM_SETUP_DIR, "bc_data.csv"), index=False)

if __name__ == "__main__":
    create_data()