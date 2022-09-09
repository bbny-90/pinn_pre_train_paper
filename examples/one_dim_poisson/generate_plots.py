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


def plot_solutiom(case:str):
    out_dir = pjoin(SCRIPT_DIR, f".tmp/{case}/")
    transp = np.linspace(0., 1., int(num_seeds*1.5))
    for i in range(num_seeds):
        data = pd.read_csv(
            pjoin(out_dir, f"solution_after_train_{i}.csv")
        )
        plt.plot(data.x, data.u, label=f'trial {i+1}', color='b', alpha=transp[i+1])
    plt.plot(data.x, get_true_solution(data.x.values), label='exact', color='r', linestyle='--')
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
    plot_solutiom("vanilla_pinn")