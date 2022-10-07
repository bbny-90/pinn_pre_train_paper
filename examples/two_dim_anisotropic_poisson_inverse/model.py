from typing import Tuple
import os
import json
import numpy as np
import torch
from models.nueral_net_pt import MLP
from pdes.operators import get_grad_scalar

class MLPSCALED(MLP):
    def __init__(self, 
            **params) -> None:
        print(params['mlp_config'])
        super().__init__(**(params['mlp_config']))
        self.x_stats = params['x_stats']
        self.u_stats = params['u_stats']
        self.ndim_x = len(self.x_stats['mean'])

        self.mean_x = torch.from_numpy(self.x_stats['mean']).float().requires_grad_(False)
        self.std_x = torch.from_numpy(self.x_stats['std']).float().requires_grad_(False)
        self.mean_u = torch.tensor(self.u_stats['mean']).float().requires_grad_(False)
        self.std_u = torch.tensor(self.u_stats['std']).float().requires_grad_(False)

        self._perm_params = torch.nn.Parameter(
            torch.from_numpy(np.random.rand(3)).float(),
            requires_grad=True
        )

    def get_perm(self):
        self.perm = torch.empty(2, 2)
        self.perm[0, 0] = self._perm_params[0]
        self.perm[1, 1] = self._perm_params[1]
        self.perm[0, 1] = self._perm_params[2]
        self.perm[1, 0] = self._perm_params[2]
        return self.perm

    def forward(self, x:torch.tensor) -> Tuple[torch.tensor, torch.tensor]:
        tmp = super().forward((x - self.mean_x) / self.std_x) # TODO: zero devision
        return  tmp * self.std_u + self.mean_u

    def calc_flux(self, u, x, perm, device=None):
        flux = get_grad_scalar(u, x, device=device)
        flux = - torch.matmul(perm, flux.unsqueeze(-1)).squeeze()
        return flux
        
    def save(self, dir_to_save: str, model_info_name: str, weight_name: str) -> None:
        super().save(dir_to_save, model_info_name, weight_name)        
        with open(os.path.join(dir_to_save, 'cord_stats.json'), "w") as f:
            tmp = {k:v.tolist() for k, v in self.x_stats.items()}
            json.dump(tmp, f)
        with open(os.path.join(dir_to_save, 'u_stats.json'), "w") as f:
            tmp = dict()
            for k, v in self.u_stats.items():
                if isinstance(v, np.ndarray):
                    tmp[k] = v.tolist()
                else:
                    tmp[k] = v
            json.dump(tmp, f)