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
from models.nueral_net_pt import MLP
from trainer.poisson.pinn_two_dim_inverse import train
from helper.other import get_torch_device


NETWORK_NAME = "MLPTEACHER"
TRAIN_NAME = "MLPTEACHER"

problem_data_dir = pjoin(SCRIPT_DIR, "data/")
out_dir = pjoin(SCRIPT_DIR, ".tmp/teachers/")


class MLPSCALED(MLP):
    def __init__(self, 
            x_stats:Dict[str, np.array],
            u_stats:Dict[str, np.array],
            params: dict, nn_weights_path=None) -> None:
        super().__init__(params, nn_weights_path)
        self.x_stats = x_stats
        self.u_stats = u_stats
        self.ndim_x = len(x_stats['mean'])

        self.mean_x = torch.from_numpy(self.x_stats['mean']).float().requires_grad_(False)
        self.std_x = torch.from_numpy(self.x_stats['std']).float().requires_grad_(False)
        self.mean_u = torch.tensor(self.u_stats['mean']).float().requires_grad_(False)
        self.std_u = torch.tensor(self.u_stats['std']).float().requires_grad_(False)

        self._perm_params = torch.nn.Parameter(
            torch.from_numpy(np.random.rand(3)).float(),
            requires_grad=True
        )

    def get_perm(self):
        self.perm = torch.empty(2, 2)
        self.perm[0, 0] = self._perm_params[0]
        self.perm[1, 1] = self._perm_params[1]
        self.perm[0, 1] = self._perm_params[2]
        self.perm[1, 0] = self._perm_params[2]
        return self.perm

    def forward(self, x:torch.tensor) -> Tuple[torch.tensor, torch.tensor]:
        tmp = super().forward((x - self.mean_x) / self.std_x) # TODO: zero devision
        return  tmp * self.std_u + self.mean_u

    def save(self, dir_to_save: str, model_info_name: str, weight_name: str) -> None:
        super().save(dir_to_save, model_info_name, weight_name)        
        with open(os.path.join(dir_to_save, 'cord_stats.json'), "w") as f:
            tmp = {k:v.tolist() for k, v in self.x_stats.items()}
            json.dump(tmp, f)
        with open(os.path.join(dir_to_save, 'u_stats.json'), "w") as f:
            tmp = {k:v.tolist() for k, v in self.u_stats.items()}
            json.dump(tmp, f)

def read_and_train(random_seed = None):
    if random_seed is not None:
        torch.manual_seed(random_seed)
    else:
        random_seed = "rand"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    dbc_data_df = pd.read_csv(pjoin(problem_data_dir, "dbc_data.csv"))
    pde_data_df = pd.read_csv(pjoin(problem_data_dir, "pde_data.csv"))
    exp_data_df = pd.read_csv(pjoin(problem_data_dir, "exp_data.csv"))
    
    data_x = dbc_data_df[['x0', 'x1']].to_numpy()
    x_stats = {'mean':data_x.mean(axis=0), 'std':data_x.std(axis=0)}
    data_u = np.concatenate([exp_data_df.u.to_numpy(), dbc_data_df.u.to_numpy()])
    u_stats = {'mean':data_u.mean(axis=0), 'std':data_u.std(axis=0)}

    with open(pjoin(SCRIPT_DIR, "configs/network.yaml")) as f:
        mlp_config = yaml.safe_load(f)[NETWORK_NAME]
    with open(pjoin(SCRIPT_DIR, "configs/train.yaml")) as f:
        train_config = yaml.safe_load(f)[TRAIN_NAME]

    device = get_torch_device()
    sol = MLPSCALED(x_stats, u_stats, mlp_config)
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
        pjoin(out_dir, f'loss_train_{random_seed}.csv'), index=False
    )
    # response after training
    # with torch.no_grad():
    #     sol.eval()
    #     xbc = bc_data_df.x.values.tolist()
    #     x = xbc[0:1] + pde_data_df.x.values.tolist() + xbc[1:2]
    #     x = torch.tensor(x).reshape(-1, 1)
    #     mixed_pred = sol(x)
    #     pd.DataFrame(
    #         {'x':x.numpy().flatten(), 'u':mixed_pred[:, 0].numpy(),'du':mixed_pred[:, 1].numpy()}
    #     ).to_csv(
    #         pjoin(out_dir, f'solution_after_train_{random_seed}.csv'), index=False)
    # sol.save(
    #     dir_to_save=out_dir,
    #     model_info_name=f"network_metadata_{random_seed}",
    #     weight_name=f"net_weight_{random_seed}"
    # )

if __name__ == "__main__":
    random_seed = None
    if len(sys.argv) > 1:
        print(sys.argv[1])
        random_seed = int(sys.argv[1])
    read_and_train(random_seed)