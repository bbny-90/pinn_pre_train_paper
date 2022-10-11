from cProfile import label
import json
import os
import pathlib
import yaml


pjoin = os.path.join
SCRIPT_DIR = os.path.abspath(pathlib.Path(__file__).parent.absolute())
PROBLEM_SETUP_DIR = pjoin(SCRIPT_DIR, "data")

import pandas as pd
import numpy as np
import torch
from helper.plotter import contour
from examples.two_dim_biharmonic.problem_setup import (
    ExactSolution,
    get_train_data_information,
    PLANE_COND,
    POISSON_RATIO,
    ELAST_MOD
)

from examples.two_dim_biharmonic.vanilla_pinn import OUT_DIR as OUT_DIR_VANILLA
from examples.two_dim_biharmonic.vanilla_pinn import NETWORK_NAME as NETWORK_NAME_VANILLA
from examples.two_dim_biharmonic.guided_pinn_pcgrad import OUT_DIR as OUT_DIR_GUIDE
from examples.two_dim_biharmonic.guided_pinn_pcgrad import NETWORK_NAME as NETWORK_NAME_GUIDE
from mechnics.solid import get_vonMisses_stress, ElasticityLinear
from examples.two_dim_biharmonic.model import MLPSCALED

import matplotlib.pyplot as plt
from matplotlib import rc
rc('font',size=18)
rc('font',family='serif')
rc('axes',labelsize=20)
rc('lines', linewidth=2,markersize=10)
# plt.rcParams["figure.figsize"] = (8,8)

def create_test_data(Nx:int, Ny:int)->np.ndarray:
    x0 = np.linspace(0., .5, Nx)
    x1 = np.linspace(0., .5, Ny)
    x0, x1 = np.meshgrid(x0, x1)
    x0 = x0.flatten().reshape(-1, 1)
    x1 = x1.flatten().reshape(-1, 1)
    x = np.hstack((x0, x1))
    return x

def plot_points(pde_points, guide_pnts, dbc_points, nbc_points):
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
    plt.savefig(pjoin(OUT_DIR_GUIDE, "domain.png"))
    # plt.show()
    plt.close()


def get_pinn_vanilla_sol(x:np.ndarray)->None:
    with open(pjoin(SCRIPT_DIR, "configs/network.yaml")) as f:
        mlp_config = yaml.safe_load(f)[NETWORK_NAME_VANILLA]
    with open(pjoin(OUT_DIR_VANILLA, "cord_stats.json"), "r") as f:
        x_stats = json.load(f)
        for k, v in x_stats.items():
            x_stats[k] = np.array(v)
    with open(pjoin(OUT_DIR_VANILLA, "u_stats.json"), "r") as f:
        u_stats = json.load(f)
        for k, v in u_stats.items():
            u_stats[k] = np.array(v)
    with open(pjoin(OUT_DIR_VANILLA, "eps_stats.json"), "r") as f:
        eps_stats = json.load(f)
        for k, v in eps_stats.items():
            eps_stats[k] = np.array(v)

    temp_params = {"mlp_config":mlp_config}
    temp_params['x_stats'] = x_stats
    temp_params['u_stats'] = u_stats
    temp_params['eps_stats'] = eps_stats
    sol = MLPSCALED(**temp_params)

    sol.load_parameters(pjoin(OUT_DIR_VANILLA, "net_weight_0.pt"))
    sol.eval()
    with torch.no_grad():
        u, eps = sol(torch.from_numpy(x).float())
        u = u.numpy()
        eps = eps.numpy()
    return u, eps

def get_pinn_guided_sol(x:np.ndarray)->None:
    with open(pjoin(SCRIPT_DIR, "configs/network.yaml")) as f:
        mlp_config = yaml.safe_load(f)[NETWORK_NAME_GUIDE]
    with open(pjoin(OUT_DIR_GUIDE, "cord_stats.json"), "r") as f:
        x_stats = json.load(f)
        for k, v in x_stats.items():
            x_stats[k] = np.array(v)
    with open(pjoin(OUT_DIR_GUIDE, "u_stats.json"), "r") as f:
        u_stats = json.load(f)
        for k, v in u_stats.items():
            u_stats[k] = np.array(v)
    with open(pjoin(OUT_DIR_GUIDE, "eps_stats.json"), "r") as f:
        eps_stats = json.load(f)
        for k, v in eps_stats.items():
            eps_stats[k] = np.array(v)

    temp_params = {"mlp_config":mlp_config}
    temp_params['x_stats'] = x_stats
    temp_params['u_stats'] = u_stats
    temp_params['eps_stats'] = eps_stats
    sol = MLPSCALED(**temp_params)

    sol.load_parameters(pjoin(OUT_DIR_GUIDE, "net_weight_0.pt"))
    sol.eval()
    with torch.no_grad():
        u, eps = sol(torch.from_numpy(x).float())
        u = u.numpy()
        eps = eps.numpy()
    return u, eps

def plot_loss_pinn_vanilla(
    ):
    data_df = pd.read_csv(pjoin(OUT_DIR_VANILLA, "loss_train_0.csv"))
    plt.rcParams["figure.figsize"] = (8,7)
    for case in ['dbc', 'pde' ,'nbc', 'compat']:
        plt.plot(data_df[case].to_numpy(), label=case)
    plt.yscale('log')
    plt.xlabel('epoch'); plt.ylabel('MSE')
    plt.legend()
    plt.tight_layout()
    plt.savefig(pjoin(OUT_DIR_VANILLA, "loss_train_0.png"))
    # plt.show()
    plt.close()

def plot_loss_pinn_guide(
    ):
    data_df = pd.read_csv(pjoin(OUT_DIR_GUIDE, "loss_train_0.csv"))
    plt.rcParams["figure.figsize"] = (8,7)
    cases = {'dbc':'dbc', 'pde':'pde' ,'nbc':'nbc', 'compat':'compat', 'u_guide':r'$u_{fem}$', 'eps_guide':r'$eps_{fem}$'}
    for case, label in cases.items():
        plt.plot(data_df[case].to_numpy(), label=label)
    plt.yscale('log')
    plt.xlabel('epoch'); plt.ylabel('MSE')
    plt.legend()
    plt.tight_layout()
    plt.savefig(pjoin(OUT_DIR_GUIDE, "loss_train_0.png"))
    # plt.show()
    plt.close()

if __name__ == "__main__":
    if not os.path.exists(OUT_DIR_VANILLA):
        os.makedirs(OUT_DIR_VANILLA)
    if not os.path.exists(OUT_DIR_GUIDE):
        os.makedirs(OUT_DIR_GUIDE)
    mat_model = ElasticityLinear(plane_cond=PLANE_COND, elast_mod=ELAST_MOD, pois_ratio=POISSON_RATIO)
    
    pde_points, dbc_points, nbc_points, guide_pnts = get_train_data_information()
    plot_points(pde_points, guide_pnts, dbc_points, nbc_points)    
    Nx_tst, Ny_tst = 101, 101
    x_test = create_test_data(Nx_tst, Ny_tst)
    sol = ExactSolution()
    disp_exact = sol.get_u(x_test)
    strs_exact = sol.get_sig(x_test)
    vm_strs_exact = get_vonMisses_stress(strs_exact, 2, PLANE_COND, POISSON_RATIO)
    u_norm_exact = np.linalg.norm(disp_exact, axis=1)
    
    u_pinn, eps_pinn = get_pinn_vanilla_sol(x_test)
    u_norm_pinn = np.linalg.norm(u_pinn, axis=1)
    sig_pinn = mat_model.get_stress(eps_pinn)
    vm_strs_pinn = get_vonMisses_stress(sig_pinn, 2, PLANE_COND, POISSON_RATIO)

    u_pinn_guide, eps_pinn_guide = get_pinn_guided_sol(x_test)
    u_norm_pinn_guide = np.linalg.norm(u_pinn_guide, axis=1)
    sig_pinn_guide = mat_model.get_stress(eps_pinn_guide)
    vm_strs_pinn_guide = get_vonMisses_stress(sig_pinn_guide, 2, PLANE_COND, POISSON_RATIO)

    mask = x_test[:,0]**2 + x_test[:, 1]**2 < 0.25**2
    resh = lambda w: w.reshape(Nx_tst, Ny_tst)
    mask_field = lambda w: resh(np.ma.array(w, mask=mask))

    contour(resh(x_test[:,0]), resh(x_test[:,1]), mask_field(u_norm_exact), 
        title=None, dir_save=OUT_DIR_GUIDE, name_save="u_exact")
    contour(resh(x_test[:,0]), resh(x_test[:,1]), mask_field(u_norm_pinn), 
        title=None, dir_save=OUT_DIR_VANILLA, name_save="u_pinn")
    contour(resh(x_test[:,0]), resh(x_test[:,1]), mask_field(u_norm_pinn_guide), 
        title=None, dir_save=OUT_DIR_GUIDE, name_save="u_pinn")


    contour(resh(x_test[:,0]), resh(x_test[:,1]), mask_field(vm_strs_exact), 
        title=None, dir_save=OUT_DIR_GUIDE, name_save="sig_exact")
    contour(resh(x_test[:,0]), resh(x_test[:,1]), mask_field(vm_strs_pinn_guide), 
        title=None, dir_save=OUT_DIR_GUIDE, name_save="sig_pinn")
    contour(resh(x_test[:,0]), resh(x_test[:,1]), mask_field(vm_strs_pinn), 
        title=None, dir_save=OUT_DIR_VANILLA, name_save="sig_pinn")
    
    plot_loss_pinn_vanilla()
    plot_loss_pinn_guide()