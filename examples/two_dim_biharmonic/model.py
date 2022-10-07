from typing import Tuple
import os
import json
import numpy as np
import torch

from models.nueral_net_pt import MLP

class MLPSCALED(MLP):
    def __init__(self, 
            **params) -> None:
        super().__init__(**params['mlp_config'])
        self.x_stats = params['x_stats']
        self.u_stats = params['u_stats']
        self.eps_stats = params['eps_stats']
        self.ndim_x = len(self.x_stats['mean'])
        self.mean_u = torch.from_numpy(self.u_stats['mean']).float().requires_grad_(False)
        self.std_u = torch.from_numpy(self.u_stats['std']).float().requires_grad_(False)
        self.mean_eps = torch.from_numpy(self.eps_stats['mean']).float().requires_grad_(False)
        self.std_eps = torch.from_numpy(self.eps_stats['std']).float().requires_grad_(False)
    
    def forward(self, x:torch.tensor) -> Tuple[torch.tensor, torch.tensor]:
        r = torch.sqrt(x[:, 0:1].pow(2) + x[:, 1:2].pow(2))
        teta = torch.atan2(x[:, 1:2], x[:, 0:1])
        x_ = torch.cat((r, teta), 1)
        tmp = super().forward(x_)
        u = tmp[:, :self.ndim_x] * self.std_u + self.mean_u
        eps = tmp[:, self.ndim_x:] * self.std_eps + self.mean_eps
        return  u, eps

    def save(self, dir_to_save: str, model_info_name: str, weight_name: str) -> None:
        super().save(dir_to_save, model_info_name, weight_name)        
        with open(os.path.join(dir_to_save, 'cord_stats.json'), "w") as f:
            tmp = dict()
            for k, v in self.x_stats.items():
                if isinstance(v, np.ndarray):
                    tmp[k] = v.tolist()
                else:
                    tmp[k] = v
            json.dump(tmp, f)
        with open(os.path.join(dir_to_save, 'u_stats.json'), "w") as f:
            tmp = dict()
            for k, v in self.u_stats.items():
                if isinstance(v, np.ndarray):
                    tmp[k] = v.tolist()
                else:
                    tmp[k] = v
            json.dump(tmp, f)
        with open(os.path.join(dir_to_save, 'eps_stats.json'), "w") as f:
            tmp = dict()
            for k, v in self.eps_stats.items():
                if isinstance(v, np.ndarray):
                    tmp[k] = v.tolist()
                else:
                    tmp[k] = v
            json.dump(tmp, f)
