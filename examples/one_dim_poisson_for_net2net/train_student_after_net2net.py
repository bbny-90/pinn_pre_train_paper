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
PROBLEM_DATA_DIR = pjoin(SCRIPT_DIR, "data/")
TEACHER_DIR = pjoin(SCRIPT_DIR, ".tmp/teachers/")
STUDENT_DIR = pjoin(SCRIPT_DIR, ".tmp/student/")

def get_best_teacher_add(root_dir:str, LOSS_WEIGHTS)->Tuple[str, float]:
    min_tot_loss = np.inf
    min_add = None
    for add in os.listdir(root_dir):
        if add.startswith("loss_train_") and add.endswith(".csv"):
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
    if not os.path.exists(STUDENT_DIR):
        os.makedirs(STUDENT_DIR)
    bc_data_df = pd.read_csv(pjoin(PROBLEM_DATA_DIR, "bc_data.csv"))
    pde_data_df = pd.read_csv(pjoin(PROBLEM_DATA_DIR, "pde_data.csv"))

    
    pjoin(SCRIPT_DIR, ".tmp/student/")

    with open(pjoin(SCRIPT_DIR, "configs/network.yaml")) as f:
        mlp_config = yaml.safe_load(f)[NETWORK_NAME]
    with open(pjoin(SCRIPT_DIR, "configs/train.yaml")) as f:
        train_config = yaml.safe_load(f)[TRAIN_NAME]
    LOSS_WEIGHTS = {
        'pde':train_config['pde_weight'], 
        'bc': train_config['bc_weight'], 
        'compat':train_config['compat_weight']
    }
    
    teacher_loss_add, teacher_score = get_best_teacher_add(TEACHER_DIR, LOSS_WEIGHTS)
    teacher_id = int((teacher_loss_add.split("_")[-1]).split(".")[0])
    print(f"best teacher id: {teacher_id}")
    teacher_weight_add = pjoin(TEACHER_DIR, f"net_weight_{teacher_id}")


    device = get_torch_device()
    sol_teacher = MLP(mlp_config, teacher_weight_add)
    # make sure it is loaded correctly
    sol_after_train_df = pd.read_csv(pjoin(TEACHER_DIR, f"solution_after_train_{teacher_id}.csv"))
    with torch.no_grad():
        sol_teacher.eval()
        u_pred = sol_teacher(torch.from_numpy(sol_after_train_df.x.values.reshape(-1, 1)).float()).numpy()
        rel_err_u = np.abs(u_pred[:, 0] - sol_after_train_df['u'].to_numpy()) / (sol_after_train_df['u'].to_numpy() + 1e-5)
        rel_err_u = rel_err_u.mean()
        assert rel_err_u < 1e-7, rel_err_u
        rel_err_du = np.abs(u_pred[:, 1] - sol_after_train_df['du'].to_numpy()) / (sol_after_train_df['du'].to_numpy() + 1e-5)
        rel_err_du = rel_err_u.mean()
        assert rel_err_du < 1e-7, rel_err_du

    # now lets widen two times!
    sol_student = sol_teacher.widen(int(mlp_config['hid_dim'])*2)
    
    # lets check if the function is still preserved
    with torch.no_grad():
        sol_student.eval()
        u_pred = sol_student(torch.from_numpy(sol_after_train_df.x.values.reshape(-1, 1)).float()).numpy()
        rel_err_u = np.abs(u_pred[:, 0] - sol_after_train_df['u'].to_numpy()) / (sol_after_train_df['u'].to_numpy() + 1e-5)
        rel_err_u = rel_err_u.mean()
        assert rel_err_u < 1e-6, rel_err_u
        rel_err_du = np.abs(u_pred[:, 1] - sol_after_train_df['du'].to_numpy()) / (sol_after_train_df['du'].to_numpy() + 1e-5)
        rel_err_du = rel_err_u.mean()
        assert rel_err_du < 1e-6, rel_err_du
    
    # response before training
    with torch.no_grad():
        sol_student.eval()
        xbc = bc_data_df.x.values.tolist()
        x = xbc[0:1] + pde_data_df.x.values.tolist() + xbc[1:2]
        x = torch.tensor(x).reshape(-1, 1)
        mixed_pred = sol_student(x)
        pd.DataFrame(
            {'x':x.numpy().flatten(), 'u':mixed_pred[:, 0].numpy(), 'du':mixed_pred[:, 1].numpy()}
        ).to_csv(
            pjoin(STUDENT_DIR, f'solution_before_train_{random_seed}.csv'), index=False)
    # train
    loss_rec = train_vanilla_mixed_form(
        sol_student, 
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
        pjoin(STUDENT_DIR, f'loss_train_{random_seed}.csv'), index=False
    )
    # response after training
    with torch.no_grad():
        sol_student.eval()
        xbc = bc_data_df.x.values.tolist()
        x = xbc[0:1] + pde_data_df.x.values.tolist() + xbc[1:2]
        x = torch.tensor(x).reshape(-1, 1)
        mixed_pred = sol_student(x)
        pd.DataFrame(
            {'x':x.numpy().flatten(), 'u':mixed_pred[:, 0].numpy(),'du':mixed_pred[:, 1].numpy()}
        ).to_csv(
            pjoin(STUDENT_DIR, f'solution_after_train_{random_seed}.csv'), index=False)
    sol_student.save(
        dir_to_save=STUDENT_DIR,
        model_info_name=f"network_metadata_{random_seed}",
        weight_name=f"net_weight_{random_seed}"
    )

if __name__ == "__main__":
    random_seed = None
    if len(sys.argv) > 1:
        print(sys.argv[1])
        random_seed = int(sys.argv[1])
    read_and_train(random_seed)