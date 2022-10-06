import os
import pathlib
import numpy as np
import torch
import pandas as pd
from torch.optim import Adam
from trainer.gradient_surgery import PCGrad

pjoin = os.path.join
SCRIPT_DIR = os.path.abspath(pathlib.Path(__file__).parent.absolute())


OUT_DIR = pjoin(SCRIPT_DIR, ".tmp/")
INIT_PARAMS = [0.5, -3.]
EPOCHS = 500000
LR = 0.001

def get_obj1(x:torch.tensor):
    obj1 = 20. * torch.log10(
            torch.max(
                torch.abs(0.5*x[:, 0] + torch.tanh(x[:, 1])),
                torch.tensor(0.000005).float()
            )
        )
    return obj1

def get_obj2(x:torch.tensor):
    obj2 = 25. * torch.log10(
            torch.max(
                torch.abs(0.5*x[:, 0] - torch.tanh(x[:, 1]) + 2.),
                torch.tensor(0.000005).float()
            )
        )
    return obj2


def solve(optim_type:str):
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)
    params = torch.nn.Parameter(torch.tensor([INIT_PARAMS]).float().view(1, -1), requires_grad=True)
    optim_base = Adam(params=[params], lr=LR)
    optim = PCGrad(optim_base)
    objectives_history = []
    params_history = []
    for i in range(EPOCHS):
        optim.zero_grad()
        objectives = [get_obj1(params)[0], get_obj2(params)[0]]
        if optim_type == 'pcgrad':
            optim.backward_surgery(objectives)
        else:
            optim.backward_regular(objectives)
        optim.step()
        with torch.no_grad():
            tot_loss = sum(objectives).item()
            objectives_history.append([o.item() for o in objectives])
            params_history.append(params.detach().numpy().flatten())
            print(i, tot_loss)
    params_history = np.array(params_history)
    objectives_history = np.array(objectives_history)
    pd.DataFrame(
        {'p0':params_history[:,0], 'p1':params_history[:,1], 
         'opbj1':objectives_history[:, 0], 'opbj2':objectives_history[:, 1]}
        ).to_csv(pjoin(OUT_DIR, f'{optim_type}.csv'), index=False)

if __name__ == "__main__":
    # solve('pcgrad')
    solve('vanilla')