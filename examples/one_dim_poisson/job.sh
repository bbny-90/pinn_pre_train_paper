#!/bin/sh
PPATH=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )

python "$PPATH/problem_setup.py"

for i in {0..9}
do
    python "$PPATH/vanilla_pinn.py" "$i"
done

for data_case in zero noisy_fem fem
do
    for i in {0..9}
    do
        python "$PPATH/guided_pinn.py" "$data_case" "$i"
    done
done

python "$PPATH/generate_plots.py"