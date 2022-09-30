import os
import sys
import pathlib
import json
pjoin = os.path.join
SCRIPT_DIR = os.path.abspath(pathlib.Path(__file__).parent.absolute())

import yaml
import pandas as pd
import numpy as np
import torch
from models.nueral_net_pt import MLP
from trainer.poisson.pinn_two_dim import train_guided
from helper.other import get_torch_device


NETWORK_NAME = "MLP2DPOISSON"
problem_data_dir = pjoin(SCRIPT_DIR, "data/")

class MLPSCALED(MLP):
    def __init__(self, 
            u_mean:float, u_std:float, 
            u_min:float, u_max:float, 
            params: dict, nn_weights_path=None) -> None:
        super().__init__(params, nn_weights_path)
        self.u_mean = u_mean
        self.u_std = u_std
        self.u_min = u_min
        self.u_max = u_max
    
    def forward(self, x):
        # x_ = x * 1.
        # x_[:, 0] = (x_[:, 0] - 0.5) * 2.
        # x_[:, 1] = (x_[:, 1] - 0.5) * 2.
        # return super().forward(x_) * self.u_std + self.u_mean
        # return super().forward(x_) * (self.u_max - self.u_min) + self.u_min
        return super().forward(x)

    def save(self, dir_to_save: str, model_info_name: str, weight_name: str) -> None:
        super().save(dir_to_save, model_info_name, weight_name)        
        tmp = {'mean':self.u_mean, 'std':self.u_std, 'min':self.u_min, 'max':self.u_max}
        with open(os.path.join(dir_to_save, 'sol_stats.json'), "w") as f:
            json.dump(tmp, f)

def read_and_train(train_conf_name, random_seed = None):
    if random_seed is not None:
        torch.manual_seed(random_seed)
    else:
        random_seed = "rand"
    dbc_data_df = pd.read_csv(pjoin(problem_data_dir, "bc_data.csv"))
    pde_data_df = pd.read_csv(pjoin(problem_data_dir, "pde_data.csv"))
    validation_data_df = pd.read_csv(pjoin(problem_data_dir, "validation_data.csv"))
    guide_data_df = pd.read_csv(pjoin(problem_data_dir, "results_fem_nodal_Nx5.csv"))
    data_u = pd.concat([pde_data_df.u, dbc_data_df.u])
    u_mean, u_std = data_u.mean(), data_u.std()
    u_min, u_max = data_u.min(), data_u.max()


    with open(pjoin(SCRIPT_DIR, "configs/network.yaml")) as f:
        mlp_config = yaml.safe_load(f)[NETWORK_NAME]
    with open(pjoin(SCRIPT_DIR, "configs/train.yaml")) as f:
        train_config = yaml.safe_load(f)[train_conf_name]
    OUT_DIR = pjoin(SCRIPT_DIR, f".tmp/fem_guide_pinn_{train_conf_name}/")
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)
    LOSS_WEIGHTS = {
        "pde":train_config['pde_weight'], 
        "bc":train_config['bc_weight'],
        "guide":train_config['guide_weight']
    }
    device = get_torch_device()
    sol = MLPSCALED(u_mean, u_std, u_min, u_max, mlp_config)
    # response before training
    with torch.no_grad():
        sol.eval()
        x_tmp = np.concatenate(
                [pde_data_df[['x0', 'x1']].values,
                dbc_data_df[['x0', 'x1']].values
                ],
                axis=0
        )
        x_tmp = torch.tensor(x_tmp).float()
        u_pred_tmp = sol(x_tmp).numpy().flatten()
        pd.DataFrame({'x0':x_tmp[:,0], 'x1':x_tmp[:,1], 'u':u_pred_tmp}).to_csv(
            pjoin(OUT_DIR, f'solution_before_train_{random_seed}.csv'), index=False
        )
    # train
    loss_rec = train_guided(
        sol, 
        x_pde=pde_data_df[['x0', 'x1']].values, 
        source_pde=pde_data_df.source.values.reshape(-1, 1),
        x_dbc=dbc_data_df[['x0', 'x1']].values, 
        u_dbc=dbc_data_df.u.values.reshape(-1, 1), 
        x_guide=guide_data_df[['x0', 'x1']].values, 
        u_guide=guide_data_df.u.values.reshape(-1, 1),
        loss_weights=LOSS_WEIGHTS, 
        train_params=train_config,
        device=device,
        x_val=validation_data_df[['x0', 'x1']].values,
        u_val=validation_data_df.u.values.reshape(-1, 1)
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
    if len(sys.argv) >= 3:
        train_conf_name = sys.argv[1]
        random_seed = int(sys.argv[2])
    read_and_train(train_conf_name, random_seed)