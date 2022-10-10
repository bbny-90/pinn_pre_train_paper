import json
import os
import pathlib
import yaml
import pandas as pd
import numpy as np
import torch


pjoin = os.path.join
SCRIPT_DIR = os.path.abspath(pathlib.Path(__file__).parent.absolute())
PROBLEM_SETUP_DIR = pjoin(SCRIPT_DIR, "data")

from helper.plotter import contour
from examples.two_dim_anisotropic_poisson_inverse.problem_setup import (
    SolverSym,
    X_LEFT, 
    X_RIGHT,
    Y_BOTTOM, 
    Y_TOP
)
from examples.two_dim_anisotropic_poisson_inverse.train_student_after_net2net import STUDENT_DIR
from examples.two_dim_anisotropic_poisson_inverse.train_student_after_net2net import get_best_teacher_add
from examples.two_dim_anisotropic_poisson_inverse.train_teacher import OUT_DIR as TEACHER_DIR
from examples.two_dim_anisotropic_poisson_inverse.train_teacher import (
    NETWORK_NAME,
    TRAIN_NAME
)
from examples.two_dim_anisotropic_poisson_inverse.problem_setup import PERM
from examples.two_dim_anisotropic_poisson_inverse.model import MLPSCALED

TRAIN_TERMS_FOR_PLOT = {
    'dbc':'dbc','pde':'pde','guide_u':'u', 'guide_q0':r'$q_x$', 'guide_q1':r'$q_y$', 'pos_def_perm':'pd'}

import matplotlib.pyplot as plt
from matplotlib import rc
rc('font',size=18)
rc('font',family='serif')
rc('axes',labelsize=20)
rc('lines', linewidth=2,markersize=10)


def create_test_data(Nx:int, Ny:int)->np.ndarray:
    x0 = np.linspace(X_LEFT, X_RIGHT, Nx)
    x1 = np.linspace(Y_BOTTOM, Y_TOP, Ny)
    x0, x1 = np.meshgrid(x0, x1)
    x0 = x0.flatten().reshape(-1, 1)
    x1 = x1.flatten().reshape(-1, 1)
    x = np.hstack((x0, x1))
    return x

def plot_exact_solution(x:np.ndarray, Nx:int, Ny:int)->None:
    symb_sol = SolverSym(PERM)
    u = symb_sol.get_solution(x).flatten()
    q = symb_sol.get_flux(x)
    resh = lambda w: w.reshape(Nx, Ny)
    contour(resh(x[:,0]), resh(x[:,1]), resh(u), 
        title=None, dir_save=STUDENT_DIR, name_save="u_exact")
    contour(resh(x[:,0]), resh(x[:,1]), resh(q[:, 0]), 
        title=None, dir_save=STUDENT_DIR, name_save="qx_exact")
    contour(resh(x[:,0]), resh(x[:,1]), resh(q[:, 1]), 
        title=None, dir_save=STUDENT_DIR, name_save="qy_exact")
    return u, q

def plot_teacher(x:np.ndarray, Nx:int, Ny:int, net_id)->None:
    with open(pjoin(TEACHER_DIR, "cord_stats.json"), "r") as f:
        x_stats = json.load(f)
        for k, v in x_stats.items():
            x_stats[k] = np.array(v)
    with open(pjoin(TEACHER_DIR, "u_stats.json"), "r") as f:
        u_stats = json.load(f)

    with open(pjoin(SCRIPT_DIR, "configs/network.yaml")) as f:
        mlp_config = yaml.safe_load(f)[NETWORK_NAME]


    temp_params = {"mlp_config":mlp_config}
    temp_params['x_stats'] = x_stats
    temp_params['u_stats'] = u_stats
    sol = MLPSCALED(**temp_params)
    teacher_weight_add = pjoin(TEACHER_DIR, f"net_weight_{net_id}")
    sol.load_parameters(teacher_weight_add)
    
    sol.eval()
    x = torch.from_numpy(x).float().requires_grad_()
    perm = sol.get_perm()
    u = sol(x)
    q = sol.calc_flux(u, x, perm).detach().numpy()
    x = x.detach().numpy()
    u = u.detach().numpy().flatten()

    resh = lambda w: w.reshape(Nx, Ny)
    contour(resh(x[:,0]), resh(x[:,1]), resh(u), 
        title=None, dir_save=TEACHER_DIR, name_save=f"u_pred_tech_{net_id}")
    contour(resh(x[:,0]), resh(x[:,1]), resh(q[:, 0]), 
        title=None, dir_save=TEACHER_DIR, name_save=f"qx_pred_tech_{net_id}")
    contour(resh(x[:,0]), resh(x[:,1]), resh(q[:, 1]), 
        title=None, dir_save=TEACHER_DIR, name_save=f"qy_pred_tech_{net_id}")
    return u, q


def plot_student(x:np.ndarray, Nx:int, Ny:int, net_id:int)->None:
    with open(pjoin(STUDENT_DIR, "cord_stats.json"), "r") as f:
        x_stats = json.load(f)
        for k, v in x_stats.items():
            x_stats[k] = np.array(v)
    with open(pjoin(STUDENT_DIR, "u_stats.json"), "r") as f:
        u_stats = json.load(f)

    with open(pjoin(SCRIPT_DIR, "configs/network.yaml")) as f:
        mlp_config = yaml.safe_load(f)[NETWORK_NAME]

    mlp_config['hid_dim'] *= 10
    temp_params = {"mlp_config":mlp_config}
    temp_params['x_stats'] = x_stats
    temp_params['u_stats'] = u_stats
    sol = MLPSCALED(**temp_params)
    student_weight_add = pjoin(STUDENT_DIR, f"net_weight_{net_id}")
    sol.load_parameters(student_weight_add)
    
    sol.eval()
    x = torch.from_numpy(x).float().requires_grad_()
    perm = sol.get_perm()
    u = sol(x)
    q = sol.calc_flux(u, x, perm).detach().numpy()
    x = x.detach().numpy()
    u = u.detach().numpy().flatten()

    resh = lambda w: w.reshape(Nx, Ny)
    contour(resh(x[:,0]), resh(x[:,1]), resh(u), 
        title=None, dir_save=STUDENT_DIR, name_save=f"u_pred_student_{net_id}")
    contour(resh(x[:,0]), resh(x[:,1]), resh(q[:, 0]), 
        title=None, dir_save=STUDENT_DIR, name_save=f"qx_pred_student_{net_id}")
    contour(resh(x[:,0]), resh(x[:,1]), resh(q[:, 1]), 
        title=None, dir_save=STUDENT_DIR, name_save=f"qy_pred_student_{net_id}")
    return u, q


def plot_loss(
    data_df:pd.DataFrame, 
    save_address: str,
    **others
    ):
    plt.rcParams["figure.figsize"] = (8,7)
    for k, v in TRAIN_TERMS_FOR_PLOT.items():
        plt.plot(data_df[k].to_numpy(), label=v)
    plt.yscale('log')
    plt.xlabel('epoch'); plt.ylabel('MSE')
    if 'ylim' in others:
        plt.ylim(others['ylim'])
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_address)
    plt.show()
    plt.close()

def plot_loss_relative(
    data_df_student:pd.DataFrame, 
    data_df_teacher:pd.DataFrame,
    save_address: str,
    ):
    plt.rcParams["figure.figsize"] = (8,7)
    base_data_series = data_df_teacher.loc[len(data_df_teacher)-1]
    for k, v in TRAIN_TERMS_FOR_PLOT.items():
        plt.plot(data_df_student[k].to_numpy()/base_data_series[k], label=v)
    plt.yscale('log')
    plt.xlabel('epoch'); plt.ylabel('relative MSE')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_address)
    # plt.show()
    plt.close()

def plot_loss_relative_zoomin(
    data_df_student:pd.DataFrame, 
    data_df_teacher:pd.DataFrame,
    save_address: str,
    ):
    plt.rcParams["figure.figsize"] = (8,7)
    base_data_series = data_df_teacher.loc[len(data_df_teacher)-1]
    for case in TRAIN_TERMS_FOR_PLOT:
        plt.plot(data_df_student[case].to_numpy()[:10]/base_data_series[case], label=case)
    plt.ylim(0.4, 2.)
    plt.yscale('log')
    plt.xlabel('epoch'); plt.ylabel('relative MSE')
    
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_address)
    # plt.show()
    plt.close()

def plot_perm(file_add:str):
    data = pd.read_csv(file_add, header=None).to_numpy()
    labels = [r'$k_11$', r'$k_22$', r'$k_12$']
    for i, col in enumerate([0, 3, 1]):
        plt.plot(data[:, col], label=labels[i])
    plt.axhline(y=1., color='k', linestyle='--', linewidth=2)
    plt.axhline(y=2., color='k', linestyle='--', linewidth=2)
    plt.tight_layout()
    plt.legend()
    tmp = file_add.split("/")
    tmp[-1] = tmp[-1][:-3] + 'png'
    tmp = '/'.join(tmp)
    plt.savefig(tmp)
    plt.close()


if __name__ == "__main__":
    Nx_tst, Ny_tst = 50, 50
    with open(pjoin(SCRIPT_DIR, "configs/train.yaml")) as f:
        train_config = yaml.safe_load(f)[TRAIN_NAME]
    teacher_loss_add, teacher_score = get_best_teacher_add(TEACHER_DIR, train_config)
    teacher_id = int((teacher_loss_add.split("_")[-1]).split(".")[0])
    print(f"best teacher id {teacher_id}")


    x_test = create_test_data(Nx_tst, Ny_tst)
    u_exact, q_exact = plot_exact_solution(x_test, Nx_tst, Ny_tst)
    u_teach, q_teach = plot_teacher(x_test,Nx_tst, Ny_tst, teacher_id)
    u_st, q_st = plot_student(x_test, Nx_tst, Ny_tst, 0)
    resh = lambda w: w.reshape(Nx_tst, Ny_tst)

    contour(resh(x_test[:,0]), resh(x_test[:,1]), resh(u_exact-u_teach), 
        title=None, dir_save=TEACHER_DIR, name_save="u_err")
    contour(resh(x_test[:,0]), resh(x_test[:,1]), resh(q_exact[:,0]-q_teach[:, 0]),
        title=None, dir_save=TEACHER_DIR, name_save="qx_err")
    contour(resh(x_test[:,0]), resh(x_test[:,1]), resh(q_exact[:,1]-q_teach[:, 1]),
        title=None, dir_save=TEACHER_DIR, name_save="qy_err")

    contour(resh(x_test[:,0]), resh(x_test[:,1]), resh(u_exact-u_st), 
        title=None, dir_save=STUDENT_DIR, name_save="u_err")
    contour(resh(x_test[:,0]), resh(x_test[:,1]), resh(q_exact[:,0]-q_st[:, 0]),
        title=None, dir_save=STUDENT_DIR, name_save="qx_err")
    contour(resh(x_test[:,0]), resh(x_test[:,1]), resh(q_exact[:,1]-q_st[:, 1]),
        title=None, dir_save=STUDENT_DIR, name_save="qy_err")
    
    best_teacher_loss_df = pd.read_csv(pjoin(TEACHER_DIR, f"loss_train_{teacher_id}.csv"))
    plot_loss(best_teacher_loss_df, pjoin(TEACHER_DIR, f"loss_train_{teacher_id}.png"), 
        **{"ylim":[1e-7, 100]}
    )
    student_loss_df = pd.read_csv(pjoin(STUDENT_DIR, f"loss_train_0.csv"))
    plot_loss(student_loss_df, pjoin(STUDENT_DIR, f"loss_train_{0}.png"),
        **{"ylim":[1e-7, 100]}
    )
    plot_loss_relative(student_loss_df, best_teacher_loss_df,
        pjoin(STUDENT_DIR, f"loss_train_relative.png")
    )
    
    add_perm_teach = pjoin(TEACHER_DIR, f"perm_history_{teacher_id}.csv")
    plot_perm(add_perm_teach)
    add_perm_st = pjoin(STUDENT_DIR, f"perm_history_{0}.csv")
    plot_perm(add_perm_st)