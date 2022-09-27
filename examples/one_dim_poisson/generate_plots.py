import os
import pathlib
from typing import Dict, List
pjoin = os.path.join
SCRIPT_DIR = os.path.abspath(pathlib.Path(__file__).parent.absolute())

import pandas as pd
from examples.one_dim_poisson.problem_setup import get_true_solution


problem_data_dir = pjoin(SCRIPT_DIR, ".tmp/problem_data/")
num_seeds = 10
LOSS_COLUMN_NAME_CHANGE = {'bc':'BC', 'pde':'PDE', 'acc':'Vald'}

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font',size=18)
rc('font',family='serif')
rc('axes',labelsize=20)
rc('lines', linewidth=2,markersize=10)


def plot_solution(case:str):
    plt.rcParams["figure.figsize"] = (8,5)
    out_dir = pjoin(SCRIPT_DIR, f".tmp/{case}/")
    transp = np.linspace(0., 1., int(num_seeds*1.5))
    for i in range(num_seeds):
        data = pd.read_csv(
            pjoin(out_dir, f"solution_after_train_{i}.csv")
        )
        plt.plot(data.x, data.u, label=f'trial {i+1}', color='b', alpha=transp[i+1])
    plt.plot(data.x, get_true_solution(data.x.values), label='exact', color='r', linestyle='--')
    if case == "zero_pinn":
        aux_data = pd.read_csv(
            pjoin(SCRIPT_DIR, f"data/zero_solution_Nx10.csv")
        )
        plt.title("guided by zeros")
    elif case == "noisy_fem_pinn":
        aux_data = pd.read_csv(
            pjoin(SCRIPT_DIR, f"data/noisy_fem_solution_Nx10.csv")
        )
        plt.title("guided by coarse noisy FEM")
    elif case == "fem_pinn":
        aux_data = pd.read_csv(
            pjoin(SCRIPT_DIR, f"data/coarse_fem_solution_Nx10.csv")
        )
        plt.title("guided by coarse FEM")
    else:
        plt.title("without guidance")
        aux_data = None
    
    if aux_data is not None:
        plt.plot(aux_data.x, aux_data.u, label='auxiliary data', color='k', marker = 'x', linestyle='none')
    plt.xlabel('x')
    plt.ylabel('u')
    # plt.legend()
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', prop={'size': 14})
    plt.tight_layout()
    plt.savefig(
        pjoin(out_dir, 'solutions.png')
    )
    plt.show()

def plot_loss(
    data_df:pd.DataFrame, 
    column_name_change:Dict[str, str],
    save_address: str,
    ):
    plt.rcParams["figure.figsize"] = (8,6)
    for case in column_name_change.keys():
        plt.plot(data_df[case].to_numpy(), label=column_name_change[case])
    plt.yscale('log')
    plt.xlabel('epoch'); plt.ylabel('MSE')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_address)
    # plt.show()
    plt.close()

def plot_best_loss(case:str):
    out_dir = pjoin(SCRIPT_DIR, f".tmp/{case}/")
    best_loss, best_id = np.inf, -1
    for i in range(num_seeds):
        data = pd.read_csv(pjoin(out_dir, f"loss_train_{i}.csv"))
        if data['acc'].iloc[-1] < best_loss:
            best_loss, best_id = data['acc'].iloc[-1], i
    best_loss_data = pd.read_csv(pjoin(out_dir, f"loss_train_{best_id}.csv"))
    save_address = pjoin(out_dir, f'loss_best_vald.png')
    if case  == 'fem_pinn':
        LOSS_COLUMN_NAME_CHANGE['guide'] = 'FEM'
    elif case  == 'noisy_fem_pinn':
        LOSS_COLUMN_NAME_CHANGE['guide'] = 'noisy FEM'
    elif case  == 'noisy_fem_pinn':
        LOSS_COLUMN_NAME_CHANGE['guide'] = 'noisy (zeros)'
    else:
        pass
    plot_loss(best_loss_data, LOSS_COLUMN_NAME_CHANGE, save_address)


if __name__ == "__main__":
    # plot_solution("vanilla_pinn")
    # plot_solution("fem_pinn")
    # plot_solution("noisy_fem_pinn")
    # plot_solution("zero_pinn")
    #
    plot_best_loss("vanilla_pinn")
    plot_best_loss("fem_pinn")
    plot_best_loss("noisy_fem_pinn")
    plot_best_loss("zero_pinn")