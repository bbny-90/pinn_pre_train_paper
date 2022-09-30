import json
import os
import pathlib
import yaml
import pandas as pd
import numpy as np


pjoin = os.path.join
SCRIPT_DIR = os.path.abspath(pathlib.Path(__file__).parent.absolute())
PROBLEM_SETUP_DIR = pjoin(SCRIPT_DIR, "data")

from helper.plotter import contour
from examples.two_dim_poisson.problem_setup import (
    SolverSym,
    X_LEFT, 
    X_RIGHT,
    Y_BOTTOM, 
    Y_TOP
)
from examples.two_dim_poisson.vanilla_pinn import OUT_DIR as OUT_DIR_VAN_PINN
from examples.two_dim_poisson.vanilla_pinn import NETWORK_NAME as NETWORK_NAME_VAN_PINN
from examples.two_dim_poisson.guided_pinn import NETWORK_NAME as NETWORK_NAME_FEM_PINN



import matplotlib.pyplot as plt
from matplotlib import rc
rc('font',size=18)
rc('font',family='serif')
rc('axes',labelsize=20)
rc('lines', linewidth=2,markersize=10)

def plot_domain():
    out_dir = os.path.dirname(OUT_DIR_VAN_PINN)
    pde_data = pd.read_csv(pjoin(PROBLEM_SETUP_DIR, "pde_data.csv"))
    bc_data = pd.read_csv(pjoin(PROBLEM_SETUP_DIR, "bc_data.csv"))
    vald_data = pd.read_csv(pjoin(PROBLEM_SETUP_DIR, "validation_data.csv"))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    plt.plot(pde_data.x0, pde_data.x1, '.', label='coll')
    plt.plot(bc_data.x0, bc_data.x1, 's', label='BC')
    plt.legend()
    plt.tight_layout()
    plt.savefig(pjoin(out_dir, "domain.png"))
    plt.close()

    plt.plot(pde_data.x0, pde_data.x1, '.', label='coll')
    plt.plot(bc_data.x0, bc_data.x1, 's', label='BC')
    plt.plot(vald_data.x0, vald_data.x1, '*', label='Valid')
    plt.legend()
    plt.tight_layout()
    plt.savefig(pjoin(out_dir, "domain_with_validation_data.png"))
    plt.close()

def create_test_data(Nx:int, Ny:int)->np.ndarray:
    x0 = np.linspace(X_LEFT, X_RIGHT, Nx)
    x1 = np.linspace(Y_BOTTOM, Y_TOP, Ny)
    x0, x1 = np.meshgrid(x0, x1)
    x0 = x0.flatten().reshape(-1, 1)
    x1 = x1.flatten().reshape(-1, 1)
    x = np.hstack((x0, x1))
    return x

def plot_exact_solution(x:np.ndarray, Nx:int, Ny:int)->None:
    symb_sol = SolverSym()
    u = symb_sol.get_solution(x)
    resh = lambda w: w.reshape(Nx, Ny)
    if not os.path.exists(OUT_DIR_VAN_PINN):
        os.makedirs(OUT_DIR_VAN_PINN)
    contour(resh(x[:,0]), resh(x[:,1]), resh(u), 
        title=None, dir_save=OUT_DIR_VAN_PINN, name_save="u_exact")

def plot_pinn_vanilla_solution(x:np.ndarray, Nx:int, Ny:int)->None:
    from models.nueral_net_pt import MLP
    import torch
    
    with open(pjoin(SCRIPT_DIR, "configs/network.yaml")) as f:
        mlp_config = yaml.safe_load(f)[NETWORK_NAME_VAN_PINN]

    sol = MLP(mlp_config, 
        nn_weights_path=pjoin(OUT_DIR_VAN_PINN, "net_weight_0.pt")
    )
    sol.eval()
    with torch.no_grad():
        u = sol(torch.from_numpy(x).float()).numpy()
    resh = lambda w: w.reshape(Nx, Ny)
    if not os.path.exists(OUT_DIR_VAN_PINN):
        os.makedirs(OUT_DIR_VAN_PINN)
    contour(resh(x[:,0]), resh(x[:,1]), resh(u), 
        title=None, dir_save=OUT_DIR_VAN_PINN, name_save="u_pred")

def plot_pinn_guided_fem_solution(x:np.ndarray, Nx:int, Ny:int, train_conf_name)->None:
    from examples.two_dim_poisson.guided_pinn import MLPSCALED
    import torch
    OUT_DIR_FEM_PINN = pjoin(SCRIPT_DIR, f".tmp/fem_guide_pinn_{train_conf_name}/")
    with open(pjoin(SCRIPT_DIR, "configs/network.yaml")) as f:
        mlp_config = yaml.safe_load(f)[NETWORK_NAME_FEM_PINN]
    with open(pjoin(OUT_DIR_FEM_PINN, "sol_stats.json"), "r") as f:
        sol_stats = json.load(f)
    sol_stats['min'] = 0.
    sol_stats['max'] = 0.

    sol = MLPSCALED(
        u_mean=float(sol_stats['mean']),
        u_std=float(sol_stats['std']),
        u_max=float(sol_stats['max']),
        u_min=float(sol_stats['min']),
        params=mlp_config, 
        nn_weights_path=pjoin(OUT_DIR_FEM_PINN, "net_weight_0.pt")
    )
    sol.eval()
    with torch.no_grad():
        u = sol(torch.from_numpy(x).float()).numpy()
    resh = lambda w: w.reshape(Nx, Ny)
    if not os.path.exists(OUT_DIR_FEM_PINN):
        os.makedirs(OUT_DIR_FEM_PINN)
    contour(resh(x[:,0]), resh(x[:,1]), resh(u), 
        title=None, dir_save=OUT_DIR_FEM_PINN, name_save="u_pred")



if __name__ == "__main__":
    plot_domain()
    Nx_tst, Ny_tst = 50, 50
    x_test = create_test_data(Nx_tst, Ny_tst)
    plot_exact_solution(x_test, Nx_tst, Ny_tst)
    plot_pinn_vanilla_solution(x_test, Nx_tst, Ny_tst)
    plot_pinn_guided_fem_solution(x_test, Nx_tst, Ny_tst, "MLP2DPOISSONGUIDED1")
    plot_pinn_guided_fem_solution(x_test, Nx_tst, Ny_tst, "MLP2DPOISSONGUIDED2")