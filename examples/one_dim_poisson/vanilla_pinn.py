import os
import sys
import pathlib
pjoin = os.path.join
SCRIPT_DIR = os.path.abspath(pathlib.Path(__file__).parent.absolute())

import yaml
import pandas as pd
import torch
from models.nueral_net_pt import MLP
from trainer.poisson.pinn_one_dim import train_vanilla
from helper.other import get_torch_device


NETWORK_NAME = "MLP1DPOISSON"
TRAIN_NAME = "MLP1DPOISSON"
LOSS_WEIGHTS = {"pde":1., "bc":1.}
problem_data_dir = pjoin(SCRIPT_DIR, "data/")
out_dir = pjoin(SCRIPT_DIR, ".tmp/vanilla_pinn/")

def read_and_train(random_seed = None):
    if random_seed is not None:
        torch.manual_seed(random_seed)
    else:
        random_seed = "rand"
    bc_data_df = pd.read_csv(pjoin(problem_data_dir, "bc_data.csv"))
    pde_data_df = pd.read_csv(pjoin(problem_data_dir, "pde_data.csv"))
    # validation_data_df = pd.read_csv(pjoin(problem_data_dir, "validation_data.csv"))


    with open(pjoin(SCRIPT_DIR, "configs/network.yaml")) as f:
        mlp_config = yaml.load(f)[NETWORK_NAME]
    with open(pjoin(SCRIPT_DIR, "configs/train.yaml")) as f:
        train_config = yaml.load(f)[TRAIN_NAME]

    device = get_torch_device()
    sol = MLP(mlp_config)
    # response before training
    with torch.no_grad():
        sol.eval()
        xbc = bc_data_df.x.values.tolist()
        x = xbc[0:1] + pde_data_df.x.values.tolist() + xbc[1:2]
        x = torch.tensor(x).reshape(-1, 1)
        u_pred = sol(x)
        pd.DataFrame({'x':x.numpy().flatten(), 'u':u_pred.numpy().flatten()}).to_csv(
            pjoin(out_dir, f'solution_before_train_{random_seed}.csv'), index=False
        )
    # train
    loss_rec = train_vanilla(sol, 
        x_pde=pde_data_df.x.values.reshape(-1, 1), 
        source_pde=pde_data_df.source.values.reshape(-1, 1),
        x_bc=bc_data_df.x.values.reshape(-1, 1), 
        u_bc=bc_data_df.u.values.reshape(-1, 1), 
        loss_weights=LOSS_WEIGHTS, 
        train_params=train_config,
        device=device,
        u_true=pde_data_df.u.values.reshape(-1, 1)
    )
    pd.DataFrame(loss_rec).to_csv(
        pjoin(out_dir, f'loss_train_{random_seed}.csv'), index=False
    )
    # response after training
    with torch.no_grad():
        sol.eval()
        xbc = bc_data_df.x.values.tolist()
        x = xbc[0:1] + pde_data_df.x.values.tolist() + xbc[1:2]
        x = torch.tensor(x).reshape(-1, 1)
        u_pred = sol(x)
        pd.DataFrame({'x':x.numpy().flatten(), 'u':u_pred.numpy().flatten()}).to_csv(
            pjoin(out_dir, f'solution_after_train_{random_seed}.csv'), index=False
        )
    sol.save(
        dir_to_save=out_dir,
        model_info_name=f"network_metadata_{random_seed}",
        weight_name=f"net_weight_{random_seed}"
    )

if __name__ == "__main__":
    random_seed = None
    if len(sys.argv) > 1:
        print(sys.argv[1])
        random_seed = int(sys.argv[1])
    read_and_train(random_seed)