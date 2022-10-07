import json
import os
import sys
import pathlib
from typing import Tuple
import random
pjoin = os.path.join
SCRIPT_DIR = os.path.abspath(pathlib.Path(__file__).parent.absolute())

import yaml
import pandas as pd
import numpy as np
import torch
from trainer.poisson.pinn_two_dim_inverse import train
from models.net_to_net import widen_net_to_net
from helper.other import get_torch_device
from examples.two_dim_anisotropic_poisson_inverse.model import MLPSCALED


from examples.two_dim_anisotropic_poisson_inverse.train_teacher import OUT_DIR as TEACHER_DIR
NETWORK_NAME = "MLPTEACHER"
TRAIN_NAME = "MLPTEACHER"
PROBLEM_DATA_DIR = pjoin(SCRIPT_DIR, "data/")
STUDENT_DIR = pjoin(SCRIPT_DIR, ".tmp/student/")

def get_best_teacher_add(root_dir:str, train_config)->Tuple[str, float]:
    loss_weight = {
        'pde':train_config['weight_pde'], 
        'dbc': train_config['weight_dbc'], 
        'guide_u':train_config['weight_guide_u'], 
        'guide_q0':train_config['weight_guide_flux0'], 
        'guide_q1':train_config['weight_guide_flux1'], 
        'pos_def_perm':train_config['weight_pos_def_perm']
    }
    min_tot_loss = np.inf
    min_add = None
    for add in os.listdir(root_dir):
        if add.startswith("loss_train_") and add.endswith(".csv"):
            new_add = pjoin(root_dir, add)
            df = pd.read_csv(new_add)
            new_score = sum([loss_weight[i] * df[i].values[-1] for i in loss_weight])
            if min_tot_loss > new_score:
                min_tot_loss = new_score
                min_add = new_add
    assert not (min_add is None), "files cannot be found"
    return new_add, min_tot_loss




def read_and_train(random_seed = None):
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
    else:
        random_seed = "rand"
    if not os.path.exists(STUDENT_DIR):
        os.makedirs(STUDENT_DIR)

    with open(pjoin(SCRIPT_DIR, "configs/network.yaml")) as f:
        mlp_config = yaml.safe_load(f)[NETWORK_NAME]
    with open(pjoin(SCRIPT_DIR, "configs/train.yaml")) as f:
        train_config = yaml.safe_load(f)[TRAIN_NAME]
    with open(pjoin(TEACHER_DIR, "cord_stats.json"), "r") as f:
        x_stats = json.load(f)
        for k, v in x_stats.items():
            x_stats[k] = np.array(v)
    with open(pjoin(TEACHER_DIR, "u_stats.json"), "r") as f:
        u_stats = json.load(f)

    teacher_loss_add, teacher_score = get_best_teacher_add(TEACHER_DIR, train_config)
    teacher_id = int((teacher_loss_add.split("_")[-1]).split(".")[0])
    print(f"best teacher id: {teacher_id}")
    teacher_weight_add = pjoin(TEACHER_DIR, f"net_weight_{teacher_id}")


    device = get_torch_device()
    temp_params = {"mlp_config":mlp_config}
    temp_params['x_stats'] = x_stats
    temp_params['u_stats'] = u_stats
    sol_teacher = MLPSCALED(**temp_params)
    sol_teacher.load_parameters(teacher_weight_add)
    # make sure it is loaded correctly
    sol_after_train_df = pd.read_csv(pjoin(TEACHER_DIR, f"solution_after_train_{teacher_id}.csv"))
    with torch.no_grad():
        sol_teacher.eval()
        u_pred = sol_teacher(torch.from_numpy(sol_after_train_df[['x0', 'x1']].values).float()).numpy().flatten()
        rel_err_u = np.abs(u_pred - sol_after_train_df['u'].to_numpy()) / (sol_after_train_df['u'].to_numpy() + 1e-5)
        rel_err_u = rel_err_u.mean()
        assert rel_err_u < 1e-7, rel_err_u

    # now lets widen two times!
    others = {"x_stats":x_stats, "u_stats":u_stats}
    sol_student:MLPSCALED = widen_net_to_net(
        old_net=sol_teacher, new_hid_dim=int(mlp_config['hid_dim'])*10, **others)
    sol_student._perm_params = sol_teacher._perm_params
    
    # lets check if the function is still preserved
    with torch.no_grad():
        sol_student.eval()
        u_pred = sol_student(torch.from_numpy(sol_after_train_df[['x0', 'x1']].to_numpy()).float()).numpy().flatten()
        rel_err_u = np.abs(u_pred - sol_after_train_df['u'].to_numpy()) / (sol_after_train_df['u'].to_numpy() + 1e-5)
        rel_err_u = rel_err_u.mean()
        assert rel_err_u < 1e-5, rel_err_u

    dbc_data_df = pd.read_csv(pjoin(PROBLEM_DATA_DIR, "dbc_data.csv"))
    pde_data_df = pd.read_csv(pjoin(PROBLEM_DATA_DIR, "pde_data.csv"))
    exp_data_df = pd.read_csv(pjoin(PROBLEM_DATA_DIR, "exp_data.csv"))
    
    data_x = dbc_data_df[['x0', 'x1']].to_numpy()
    x_stats = {'mean':data_x.mean(axis=0), 'std':data_x.std(axis=0)}
    data_u = np.concatenate([exp_data_df.u.to_numpy(), dbc_data_df.u.to_numpy()])
    u_stats = {'mean':data_u.mean(axis=0), 'std':data_u.std(axis=0)}
    
    # train
    loss_rec = train(
        solution=sol_student,
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
        pjoin(STUDENT_DIR, f'loss_train_{random_seed}.csv'), index=False
    )
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