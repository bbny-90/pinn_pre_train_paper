import random
import os
import sys
import pathlib
pjoin = os.path.join
SCRIPT_DIR = os.path.abspath(pathlib.Path(__file__).parent.absolute())

import yaml
import pandas as pd
import torch
import numpy as np
from trainer.poisson.pinn_two_dim_inverse import train
from examples.two_dim_anisotropic_poisson_inverse.model import MLPSCALED
from helper.other import get_torch_device


NETWORK_NAME = "MLPTEACHER"
TRAIN_NAME = "MLPTEACHER"

PROBLEM_DATA_DIR = pjoin(SCRIPT_DIR, "data/")
OUT_DIR = pjoin(SCRIPT_DIR, ".tmp/teachers/")


def read_and_train(random_seed = None):
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
    else:
        random_seed = "rand"
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)
    dbc_data_df = pd.read_csv(pjoin(PROBLEM_DATA_DIR, "dbc_data.csv"))
    pde_data_df = pd.read_csv(pjoin(PROBLEM_DATA_DIR, "pde_data.csv"))
    exp_data_df = pd.read_csv(pjoin(PROBLEM_DATA_DIR, "exp_data.csv"))
    
    data_x = dbc_data_df[['x0', 'x1']].to_numpy()
    x_stats = {'mean':data_x.mean(axis=0), 'std':data_x.std(axis=0)}
    data_u = np.concatenate([exp_data_df.u.to_numpy(), dbc_data_df.u.to_numpy()])
    u_stats = {'mean':data_u.mean(axis=0), 'std':data_u.std(axis=0)}

    with open(pjoin(SCRIPT_DIR, "configs/network.yaml")) as f:
        mlp_config = yaml.safe_load(f)[NETWORK_NAME]
    with open(pjoin(SCRIPT_DIR, "configs/train.yaml")) as f:
        train_config = yaml.safe_load(f)[TRAIN_NAME]

    device = get_torch_device()
    tmp_params = {"mlp_config":mlp_config}
    tmp_params['x_stats'] = x_stats
    tmp_params['u_stats'] = u_stats
    sol = MLPSCALED(**tmp_params)
    # train
    loss_rec = train(
        solution=sol,
        x_pde = pde_data_df[['x0', 'x1']].to_numpy(),
        source_pde = pde_data_df['source'].to_numpy().reshape(-1, 1),
        x_dbc = dbc_data_df[['x0', 'x1']].to_numpy(),
        u_dbc = dbc_data_df['u'].to_numpy().reshape(-1, 1),
        x_guide = exp_data_df[['x0', 'x1']].to_numpy(),
        u_guide = exp_data_df['u'].to_numpy().reshape(-1, 1),
        flux_guide = exp_data_df[['flux0', 'flux1']].to_numpy(),
        train_params = train_config,
        device = device,
    )
    pd.DataFrame(loss_rec).to_csv(
        pjoin(OUT_DIR, f'loss_train_{random_seed}.csv'), index=False
    )
    sol.save(
        dir_to_save=OUT_DIR,
        model_info_name=f"network_metadata_{random_seed}",
        weight_name=f"net_weight_{random_seed}"
    )
    with torch.no_grad():
        sol.eval()
        x = torch.from_numpy(pde_data_df[['x0', 'x1']].to_numpy()).float()
        u = sol(x).numpy().flatten()
        pd.DataFrame(
            {'x0':pde_data_df.x0, 'x1':pde_data_df.x1, 'u':u}
        ).to_csv(
            pjoin(OUT_DIR, f'solution_after_train_{random_seed}.csv'), index=False)

if __name__ == "__main__":
    random_seed = None
    if len(sys.argv) > 1:
        print(sys.argv[1])
        random_seed = int(sys.argv[1])
    read_and_train(random_seed)