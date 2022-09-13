import os
import pathlib
import numpy as np
import pandas as pd

pjoin = os.path.join
SCRIPT_DIR = os.path.abspath(pathlib.Path(__file__).parent.absolute())
PROBLEM_SETUP_DIR = pjoin(SCRIPT_DIR, "data")
RANDOM_STATE = 0
X_LEFT, X_RIGHT = -10., 10.
NUM_VALD_PNTS = 50
NUM_PDE_PNTS = 60

def get_bc_pnts(x_lim, y_lim, n, debug=False):
    x_bc, y_bc = [], []
    # bottom
    x_bc.append(np.linspace(x_lim[0], x_lim[1], n))
    y_bc.append(np.ones_like(x_bc[-1])*y_lim[0])
    # top
    x_bc.append(np.linspace(x_lim[0], x_lim[1], n))
    y_bc.append(np.ones_like(x_bc[-1])*y_lim[1])
    # left
    y_bc.append(np.linspace(y_lim[0], y_lim[1], n, endpoint=False)[1:])
    x_bc.append(np.ones_like(y_bc[-1])*x_lim[0])
    # right
    y_bc.append(np.linspace(y_lim[0], y_lim[1], n, endpoint=False)[1:])
    x_bc.append(np.ones_like(y_bc[-1])*x_lim[1])
    # gather
    x_bc = np.concatenate(x_bc)
    y_bc = np.concatenate(y_bc)
    xy_bc = np.hstack((x_bc.reshape(-1, 1), y_bc.reshape(-1, 1)))
    return xy_bc

def get_exact_sol(x, y):
    w = 4*np.pi
    return np.sin(w*y)*np.exp(-w*x)

def get_f(x_in, y_in):
    import sympy as sp 
    from sympy.abc import x, y
    w = 4*sp.pi
    u = sp.sin(w*y)*sp.exp(-w*x)
    du_dx = sp.diff(u, x)
    ddu_dxx = sp.diff(du_dx, x)
    du_dy = sp.diff(u, y)
    ddu_dyy = sp.diff(du_dy, y)
    div = ddu_dxx + ddu_dyy
    f = sp.lambdify([x, y], div, "numpy")
    val = f(x_in, y_in)
    if isinstance(val, int) or isinstance(val, float):
        print(20*"!")
        val = val * np.ones_like(x_in)
    return val

def create_data():
    if not os.path.exists(PROBLEM_SETUP_DIR):
        os.makedirs(PROBLEM_SETUP_DIR)
    rand = np.random.RandomState(RANDOM_STATE)
    x_pde = np.linspace(X_LEFT, X_RIGHT, NUM_PDE_PNTS, endpoint=False)[1:]
    x_valid = rand.uniform(X_LEFT, X_RIGHT, NUM_VALD_PNTS)
    x_bc = np.array([X_LEFT, X_RIGHT])
    
    u_pde = get_true_solution(x_pde)
    source_pde = get_true_sorce(x_pde)

    u_valid = get_true_solution(x_valid)
    source_valid = get_true_sorce(x_valid)

    u_bc = get_true_solution(x_bc)

    
    pd.DataFrame({'x':x_pde, 'u':u_pde, 'source':source_pde}
    ).to_csv(pjoin(PROBLEM_SETUP_DIR, "pde_data.csv"), index=False)
    pd.DataFrame({'x':x_valid, 'u':u_valid, 'source':source_valid}
    ).to_csv(pjoin(PROBLEM_SETUP_DIR, "validation_data.csv"), index=False)
    pd.DataFrame({'x':x_bc, 'u':u_bc}
    ).to_csv(pjoin(PROBLEM_SETUP_DIR, "bc_data.csv"), index=False)
    
def create_noise_data():
    fem_data = pd.read_csv(
        pjoin(SCRIPT_DIR, "data/coarse_fem_solution_Nx10.csv")
    )
    rand = np.random.RandomState(RANDOM_STATE)
    u_noisy = fem_data.u.values + rand.randn(9) * 2.
    u_zero = np.zeros_like(u_noisy)
    pd.DataFrame({"x":fem_data.x.values.flatten(), "u":u_noisy.flatten()}).to_csv(
        pjoin(SCRIPT_DIR, "data/noisy_fem_solution_Nx10.csv"), index=False
    )
    pd.DataFrame({"x":fem_data.x.values.flatten(), "u":u_zero.flatten()}).to_csv(
        pjoin(SCRIPT_DIR, "data/zero_solution_Nx10.csv"), index=False
    )
    

if __name__ == "__main__":
    create_data()
    create_noise_data()