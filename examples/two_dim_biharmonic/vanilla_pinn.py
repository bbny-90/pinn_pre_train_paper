import os
import sys
import pathlib
import random
pjoin = os.path.join
SCRIPT_DIR = os.path.abspath(pathlib.Path(__file__).parent.absolute())

import yaml
import pandas as pd
import torch
import numpy as np
from mechnics.solid import ElasticityLinear
from trainer.biharmonic.pinn_two_dim import train_guided_mixed_disp_strain
from helper.other import get_torch_device
from examples.two_dim_biharmonic.problem_setup import (
    PLANE_COND,
    ELAST_MOD,
    POISSON_RATIO,
    get_train_data_information
)
from examples.two_dim_biharmonic.model import MLPSCALED
NETWORK_NAME = "MLP2DBYHARMONIC"
TRAIN_NAME = "MLP2DBYHARMONICG"
problem_data_dir = pjoin(SCRIPT_DIR, "data/")
OUT_DIR = pjoin(SCRIPT_DIR, ".tmp/vannila_pinn/")


def read_and_train(random_seed = None):
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
    else:
        random_seed = "rand"
    pde_points, dbc_points, nbc_points, _ = get_train_data_information()
    guide_pnts = []
    # if 1:
    #     plot_points(pde_points, guide_pnts, dbc_points, nbc_points)
    data_x = [i.x for i in pde_points]
    data_x = np.concatenate(data_x)
    
    x_stats = {'mean':data_x.mean(axis=0), 'std':data_x.std(axis=0)}
    u_stats = {'mean':np.zeros(2), 'std':np.ones(2)}
    eps_stats = {'mean':np.zeros(3), 'std':np.ones(3)}
    

    with open(pjoin(SCRIPT_DIR, "configs/network.yaml")) as f:
        mlp_config = yaml.full_load(f)[NETWORK_NAME]
    with open(pjoin(SCRIPT_DIR, "configs/train.yaml")) as f:
        train_config = yaml.full_load(f)[TRAIN_NAME]
    mat_model = ElasticityLinear(plane_cond=PLANE_COND, elast_mod=ELAST_MOD, pois_ratio=POISSON_RATIO)
    device = get_torch_device()
    tmp_params = {'mlp_config':mlp_config}
    tmp_params['x_stats'] = x_stats
    tmp_params['u_stats'] = u_stats
    tmp_params['eps_stats'] = eps_stats
    sol = MLPSCALED(**tmp_params)
    # train
    loss_rec = train_guided_mixed_disp_strain(
        solution = sol,
        mat_model= mat_model,
        pde_pnts=pde_points,
        dbc_pnts=dbc_points,
        nbc_pnts=nbc_points,
        guide_pnts=guide_pnts,
        #
        train_params=train_config,
        device=device,
        #
        x_val = None, 
        source_pde_val = None,
    )

    pd.DataFrame(loss_rec).to_csv(
        pjoin(OUT_DIR, f'loss_train_{random_seed}.csv'), index=False
    )
    sol.save(
        dir_to_save=OUT_DIR,
        model_info_name=f"network_metadata_{random_seed}",
        weight_name=f"net_weight_{random_seed}.pt"
    )

if __name__ == "__main__":
    random_seed = None
    if len(sys.argv) > 1:
        print(sys.argv[1])
        random_seed = int(sys.argv[1])
    read_and_train(random_seed)