#!/bin/sh
PPATH=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )

for i in {0..49}
do
    python "$PPATH/train_teacher.py" "$i"
done
