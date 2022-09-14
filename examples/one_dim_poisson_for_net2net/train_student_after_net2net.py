import os
import sys
import pathlib
from typing import Tuple
pjoin = os.path.join
SCRIPT_DIR = os.path.abspath(pathlib.Path(__file__).parent.absolute())

import yaml
import pandas as pd
import numpy as np
import torch
from models.nueral_net_pt import MLP
from trainer.poisson.pinn_one_dim import train_vanilla_mixed_form
from helper.other import get_torch_device


NETWORK_NAME = "MLP1DPOISSONTEACHER"
TRAIN_NAME = "MLP1DPOISSONSTUDENT"
LOSS_WEIGHTS = {"pde":1., "bc":1., "compat":1.}
problem_data_dir = pjoin(SCRIPT_DIR, "data/")
teacher_dir = pjoin(SCRIPT_DIR, ".tmp/teachers/")
out_dir = pjoin(SCRIPT_DIR, ".tmp/student/")

def get_best_teacher_add(root_dir:str)->Tuple[str, float]:
    min_tot_loss = np.inf
    min_add = None
    for add in os.listdir(root_dir):
        if add.startswith("loss_train_"):
            new_add = pjoin(root_dir, add)
            df = pd.read_csv(new_add)
            new_score = sum([LOSS_WEIGHTS[i] * df[i].values[-1] for i in LOSS_WEIGHTS])
            if min_tot_loss > new_score:
                min_tot_loss = new_score
                min_add = new_add
    assert not (min_add is None), "files cannot be found"
    return new_add, min_tot_loss




def read_and_train(random_seed = None):
    if random_seed is not None:
        torch.manual_seed(random_seed)
    else:
        random_seed = "rand"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    bc_data_df = pd.read_csv(pjoin(problem_data_dir, "bc_data.csv"))
    pde_data_df = pd.read_csv(pjoin(problem_data_dir, "pde_data.csv"))

    
    pjoin(SCRIPT_DIR, ".tmp/student/")

    with open(pjoin(SCRIPT_DIR, "configs/network.yaml")) as f:
        mlp_config = yaml.load(f)[NETWORK_NAME]
    with open(pjoin(SCRIPT_DIR, "configs/train.yaml")) as f:
        train_config = yaml.load(f)[TRAIN_NAME]

    teacher_loss_add, teacher_score = get_best_teacher_add(teacher_dir)
    teacher_id = int((teacher_loss_add.split("_")[-1]).split(".")[0])
    teacher_weight_add = pjoin(teacher_dir, f"net_weight_{teacher_id}")


    device = get_torch_device()
    sol = MLP(mlp_config, teacher_weight_add)
    # make sure it is loaded correctly
    sol_after_train_df = pd.read_csv(pjoin(teacher_dir, f"solution_after_train_{teacher_id}.csv"))
    with torch.no_grad():
        sol.eval()
        u_pred = sol(torch.from_numpy(sol_after_train_df.x.values.reshape(-1, 1)).float()).numpy()
        assert np.allclose(u_pred, sol_after_train_df[['u', 'du']].values)
    # now lets widen two times!
    sol.widen(int(mlp_config['hid_dim'])*2)
    # lets check if the function is still preserved
    with torch.no_grad():
        sol.eval()
        u_pred = sol(torch.from_numpy(sol_after_train_df.x.values.reshape(-1, 1)).float()).numpy()
        assert np.allclose(u_pred, sol_after_train_df[['u', 'du']].values)
    
    # response before training
    with torch.no_grad():
        sol.eval()
        xbc = bc_data_df.x.values.tolist()
        x = xbc[0:1] + pde_data_df.x.values.tolist() + xbc[1:2]
        x = torch.tensor(x).reshape(-1, 1)
        mixed_pred = sol(x)
        pd.DataFrame(
            {'x':x.numpy().flatten(), 'u':mixed_pred[:, 0].numpy(), 'du':mixed_pred[:, 1].numpy()}
        ).to_csv(
            pjoin(out_dir, f'solution_before_train_{random_seed}.csv'), index=False)
    # train
    loss_rec = train_vanilla_mixed_form(
        sol, 
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
        mixed_pred = sol(x)
        pd.DataFrame(
            {'x':x.numpy().flatten(), 'u':mixed_pred[:, 0].numpy(),'du':mixed_pred[:, 1].numpy()}
        ).to_csv(
            pjoin(out_dir, f'solution_after_train_{random_seed}.csv'), index=False)
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