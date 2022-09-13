import os
import pathlib
pjoin = os.path.join
SCRIPT_DIR = os.path.abspath(pathlib.Path(__file__).parent.absolute())

import yaml
import pandas as pd
import numpy as np
import torch
from models.nueral_net_pt import MLP
from helper.symbolic_calculator import EvalNp
from examples.two_dim_poisson.problem_setup import (
    SymbSol,
    X_LEFT, X_RIGHT,
    Y_BOTTOM, Y_TOP
)

NX_TEST, NY_TEST = 50, 50

def get_test_data()->np.ndarray:
    x = np.linspace(X_LEFT, X_RIGHT, NX_TEST)
    y = np.linspace(Y_BOTTOM, Y_TOP, NY_TEST)
    x, y = np.meshgrid(x, y)
    x = x.flatten().reshape(-1, 1)
    y = y.flatten().reshape(-1, 1)
    return np.hstack((x, y))

def get_vanila_pinn_pred(x:np.ndarray, u_true:np.ndarray):
    from examples.two_dim_poisson.vanilla_pinn import NETWORK_NAME
    
    with open(pjoin(SCRIPT_DIR, "configs/network.yaml")) as f:
        mlp_config = yaml.load(f)[NETWORK_NAME]
    vanilla_dir = pjoin(SCRIPT_DIR, ".tmp/vanilla_pinn")
    sol = MLP(mlp_config, 
        nn_weights_path=pjoin(vanilla_dir, "net_weight_rand")
    )
    with torch.no_grad():
        sol.eval()
        u_pred = sol(torch.from_numpy(x).float()).numpy().flatten()
    pd.DataFrame(
        {"x0":x[:, 0], "x1":x[:,1], "u_true":u_true.flatten(), "u_pred":u_pred}
    ).to_csv(pjoin(vanilla_dir, "test_data_dolution.csv"), index=False)


if __name__ == "__main__":
    x = get_test_data()
    symb_sol = SymbSol()
    eval_np_u = EvalNp(symb_sol.x, symb_sol.u, 1)
    u = eval_np_u(x)
    get_vanila_pinn_pred(x, u)