import os
import pathlib
import pandas as pd


pjoin = os.path.join
SCRIPT_DIR = os.path.abspath(pathlib.Path(__file__).parent.absolute())
PROBLEM_SETUP_DIR = pjoin(SCRIPT_DIR, "data")
OUT_DIR = pjoin(SCRIPT_DIR, ".tmp")

import matplotlib.pyplot as plt
from matplotlib import rc
rc('font',size=18)
rc('font',family='serif')
rc('axes',labelsize=20)
rc('lines', linewidth=2,markersize=10)

def plot_domain():
    pde_data = pd.read_csv(pjoin(PROBLEM_SETUP_DIR, "pde_data.csv"))
    bc_data = pd.read_csv(pjoin(PROBLEM_SETUP_DIR, "bc_data.csv"))
    vald_data = pd.read_csv(pjoin(PROBLEM_SETUP_DIR, "validation_data.csv"))
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)
    
    plt.plot(pde_data.x0, pde_data.x1, '.', label='coll')
    plt.plot(bc_data.x0, bc_data.x1, 's', label='BC')
    plt.legend()
    plt.tight_layout()
    plt.savefig(pjoin(OUT_DIR, "domain.png"))
    plt.show()
    plt.close()

    plt.plot(pde_data.x0, pde_data.x1, '.', label='coll')
    plt.plot(bc_data.x0, bc_data.x1, 's', label='BC')
    plt.plot(vald_data.x0, vald_data.x1, '*', label='Valid')
    plt.legend()
    plt.tight_layout()
    plt.savefig(pjoin(OUT_DIR, "domain_with_validation_data.png"))
    plt.show()
    plt.close()


if __name__ == "__main__":
    plot_domain()