#!/bin/sh
PPATH=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )

python "$PPATH/problem_setup.py"
python "$PPATH/vanilla_pinn.py" 0
python "$PPATH/guided_pinn.py" MLP2DPOISSONGUIDED1 0
python "$PPATH/guided_pinn.py" MLP2DPOISSONGUIDED2 0
python "$PPATH/plot_results.py" 0
