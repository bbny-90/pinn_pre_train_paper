import os
import pathlib
pjoin = os.path.join
SCRIPT_DIR = os.path.abspath(pathlib.Path(__file__).parent.absolute())

import pandas as pd
from examples.one_dim_poisson.problem_setup import get_true_solution


problem_data_dir = pjoin(SCRIPT_DIR, ".tmp/problem_data/")
num_seeds = 10

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font',size=18)
rc('font',family='serif')
rc('axes',labelsize=20)
rc('lines', linewidth=2,markersize=10)
plt.rcParams["figure.figsize"] = (8,5)


def plot_solution(case:str):
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

if __name__ == "__main__":
    plot_solution("vanilla_pinn")
    plot_solution("fem_pinn")
    plot_solution("noisy_fem_pinn")
    plot_solution("zero_pinn")