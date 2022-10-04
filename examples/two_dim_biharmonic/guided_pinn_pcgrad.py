from typing import Dict, Tuple
import os
import sys
import pathlib
import json
pjoin = os.path.join
SCRIPT_DIR = os.path.abspath(pathlib.Path(__file__).parent.absolute())

import yaml
import pandas as pd
import torch
import numpy as np
from mechnics.solid import ElasticityLinear
from models.nueral_net_pt import MLP
from trainer.biharmonic.pinn_two_dim import train_guided_mixed_disp_strain
from helper.other import get_torch_device
from examples.two_dim_biharmonic.problem_setup import (
    PLANE_COND,
    ELAST_MOD,
    POISSON_RATIO,
    get_train_data_information
)

NETWORK_NAME = "MLP2DBYHARMONIC"
TRAIN_NAME = "MLP2DBYHARMONICGUIDED"
problem_data_dir = pjoin(SCRIPT_DIR, "data/")
out_dir = pjoin(SCRIPT_DIR, ".tmp/fem_guided_pinn_pcgrad/")

class MLPSCALED(MLP):
    def __init__(self, 
            x_stats:Dict[str, np.array],
            u_stats:Dict[str, np.array],
            eps_stats:Dict[str, np.array],
            params: dict, nn_weights_path=None) -> None:
        super().__init__(params, nn_weights_path)
        self.x_stats = x_stats
        self.u_stats = u_stats
        self.eps_stats = eps_stats
        self.ndim_x = len(x_stats['mean'])
        self.mean_u = torch.from_numpy(self.u_stats['mean']).float().requires_grad_(False)
        self.std_u = torch.from_numpy(self.u_stats['std']).float().requires_grad_(False)
        self.mean_eps = torch.from_numpy(self.eps_stats['mean']).float().requires_grad_(False)
        self.std_eps = torch.from_numpy(self.eps_stats['std']).float().requires_grad_(False)
    
    def forward(self, x:torch.tensor) -> Tuple[torch.tensor, torch.tensor]:
        r = torch.sqrt(x[:, 0:1].pow(2) + x[:, 1:2].pow(2))
        teta = torch.atan2(x[:, 1:2], x[:, 0:1])
        x_ = torch.cat((r, teta), 1)
        tmp = super().forward(x_)
        u = tmp[:, :self.ndim_x] * self.std_u + self.mean_u
        eps = tmp[:, self.ndim_x:] * self.std_eps + self.mean_eps
        return  u, eps

    def save(self, dir_to_save: str, model_info_name: str, weight_name: str) -> None:
        super().save(dir_to_save, model_info_name, weight_name)        
        with open(os.path.join(dir_to_save, 'cord_stats.json'), "w") as f:
            tmp = {k:v.tolist() for k, v in self.x_stats.items()}
            json.dump(tmp, f)
        with open(os.path.join(dir_to_save, 'u_stats.json'), "w") as f:
            tmp = {k:v.tolist() for k, v in self.u_stats.items()}
            json.dump(tmp, f)
        with open(os.path.join(dir_to_save, 'eps_stats.json'), "w") as f:
            tmp = {v.tolist() for k, v in self.eps_stats.items()}
            json.dump(tmp, f)

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
    sol = MLPSCALED(x_stats, u_stats, eps_stats, mlp_config)
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