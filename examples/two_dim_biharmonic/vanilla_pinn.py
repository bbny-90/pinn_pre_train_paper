import os
import sys
import pathlib
pjoin = os.path.join
SCRIPT_DIR = os.path.abspath(pathlib.Path(__file__).parent.absolute())

import yaml
import pandas as pd
import numpy as np
import torch
from mechnics.solid import ElasticityLinear
from models.nueral_net_pt import MLP
from trainer.biharmonic.pinn_two_dim import train_vanilla_mixed_disp_strain
from helper.other import get_torch_device
from examples.two_dim_biharmonic.problem_setup import (
    PLANE_COND,
    ELAST_MOD,
    POISSON_RATIO
)

NETWORK_NAME = "MLP2DBYHARMONIC"
TRAIN_NAME = "MLP2DBYHARMONIC"
LOSS_WEIGHTS = {"pde":1., "dbc":1., "nbc":1., "compat":1.}
problem_data_dir = pjoin(SCRIPT_DIR, "data/")
out_dir = pjoin(SCRIPT_DIR, ".tmp/vanilla_pinn/")

def read_and_train(random_seed = None):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    if random_seed is not None:
        torch.manual_seed(random_seed)
    else:
        random_seed = "rand"
    dbc_data_train_df = pd.read_csv(pjoin(problem_data_dir, "dirichlet_bc_data_train.csv"))
    nbc_data_train_df = pd.read_csv(pjoin(problem_data_dir, "neumann_bc_data_train.csv"))    
    pde_data_train_df = pd.read_csv(pjoin(problem_data_dir, "pde_data_train.csv"))


    with open(pjoin(SCRIPT_DIR, "configs/network.yaml")) as f:
        mlp_config = yaml.full_load(f)[NETWORK_NAME]
    with open(pjoin(SCRIPT_DIR, "configs/train.yaml")) as f:
        train_config = yaml.full_load(f)[TRAIN_NAME]
    mat_model = ElasticityLinear(plane_cond=PLANE_COND, elast_mod=ELAST_MOD, pois_ratio=POISSON_RATIO)
    device = get_torch_device()
    sol = MLP(mlp_config)
    # train
    loss_rec = train_vanilla_mixed_disp_strain(
        solution = sol,
        mat_model= mat_model,
        x_pde = pde_data_train_df[['x0', 'x1']].values,
        source_pde = pde_data_train_df[['source0', 'source1']].values,
        #
        x_dbc = dbc_data_train_df[['x0', 'x1']].values,
        u_dbc = dbc_data_train_df[['u0', 'u1']].values,
        #
        x_nbc = nbc_data_train_df[['x0', 'x1']].values,
        n_nbc = nbc_data_train_df[['n0', 'n1']].values,
        trac_nbc = nbc_data_train_df[['trac0', 'trac1']].values,
        #
        loss_weights = LOSS_WEIGHTS,
        train_params=train_config,
        device=device,
        #
        x_val = None, 
        source_pde_val = None,
    )

    pd.DataFrame(loss_rec).to_csv(
        pjoin(out_dir, f'loss_train_{random_seed}.csv'), index=False
    )
    sol.save(
        dir_to_save=out_dir,
        model_info_name=f"network_metadata_{random_seed}",
        weight_name=f"net_weight_{random_seed}.pt"
    )

if __name__ == "__main__":
    random_seed = None
    if len(sys.argv) > 1:
        print(sys.argv[1])
        random_seed = int(sys.argv[1])
    read_and_train(random_seed)