import os
import sys
import pathlib
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
TRAIN_NAME = "MLP2DBYHARMONICGUIDED"
problem_data_dir = pjoin(SCRIPT_DIR, "data/")
out_dir = pjoin(SCRIPT_DIR, ".tmp/fem_guided_pinn_pcgrad/")


def plot_points(pde_points, guide_pnts, dbc_points, nbc_points):
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    pnts_all = []
    for pnts in pde_points:
        pnts_all.append(pnts.x)
    pnts_all = np.concatenate(pnts_all)
    plt.plot(pnts_all[:, 0], pnts_all[:, 1], '.', label='col')

    pnts_all = []
    for pnts in guide_pnts:
        pnts_all.append(pnts.x)
    pnts_all = np.concatenate(pnts_all)
    plt.plot(pnts_all[:, 0], pnts_all[:, 1], '.', label='fem')

    pnts_all = []
    for pnts in dbc_points:
        pnts_all.append(pnts.x)
    pnts_all = np.concatenate(pnts_all)
    plt.plot(pnts_all[:,0], pnts_all[:,1], 's', label='dbc')

    pnts_all = []
    for pnts in nbc_points:
        pnts_all.append(pnts.x)
    pnts_all = np.concatenate(pnts_all)
    plt.plot(pnts_all[:, 0], pnts_all[:, 1], 's', label='nbc')
    ax.set_aspect('equal', adjustable='box')
    plt.legend()
    plt.tight_layout()
    # plt.savefig(root_folder + "domain.png")
    plt.show()

def read_and_train(random_seed = None):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    if random_seed is not None:
        torch.manual_seed(random_seed)
    else:
        random_seed = "rand"
    pde_points, dbc_points, nbc_points, guide_pnts = get_train_data_information()
    if 1:
        plot_points(pde_points, guide_pnts, dbc_points, nbc_points)
    data_u = [i.disp for i in guide_pnts] + [i.val for i in dbc_points]
    data_u = np.concatenate(data_u)
    data_x = [i.x for i in guide_pnts] + [i.x for i in dbc_points]+\
            [i.x for i in pde_points] + [i.x for i in nbc_points]
    data_x = np.concatenate(data_x)
    data_eps = np.concatenate([i.strain for i in guide_pnts])
    
    x_stats = {'mean':data_x.mean(axis=0), 'std':data_x.std(axis=0)}
    u_stats = {'mean':data_u.mean(axis=0), 'std':data_u.std(axis=0)}
    eps_stats = {'mean':data_eps.mean(axis=0), 'std':data_eps.std(axis=0)}
    

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