#!/bin/sh
PPATH=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )

python "$PPATH/problem_setup.py"

for i in {0..49}
do
    python "$PPATH/train_teacher.py" "$i"
done

python "$PPATH/train_student_after_net2net.py" 0

python "$PPATH/generate_plots.py"