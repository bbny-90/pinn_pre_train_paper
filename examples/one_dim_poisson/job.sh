#!/bin/sh
PPATH=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )

# for i in {0..9}
# do
#     python "$PPATH/vanilla_pinn.py" "$i"
# done

for data_case in noisy_fem fem zero
do
    for i in {0..9}
    do
        python "$PPATH/guided_pinn.py" "$data_case" "$i"
    done
done

