# from __future__ import annotations
from typing import Dict, List, Optional
import os
import json
import torch
import numpy as np

from models.activation import SiLU
from helper.other import drop_file_type

class MLP(torch.nn.Module):
    def __init__(self, **params) -> None:
        """
            inputs:
                params = {"in_dim":
                          "out_dim":
                          "hid_dim":
                          "num_hid_layer":
                          "act_type":
                }
            ------
            e.g., a network with 3 hidden layers: in->h1->h2->h3->out
        """
        super().__init__()
        self.in_dim = int(params["in_dim"])
        self.out_dim = int(params["out_dim"])
        self.hid_dim = int(params["hid_dim"])
        self.num_hid_layer = int(params["num_hid_layer"])
        self.act_type: str = params["act_type"]
        if self.act_type == "relu":
            self.actFun = torch.nn.ReLU()
        elif self.act_type == "tanh":
            self.actFun = torch.nn.Tanh()
        elif self.act_type == "silu":
            self.actFun = SiLU()
        else:
            raise NotImplementedError(f"activation {self.act_type} is not supported")
        if "weight_initilization" in params:
            params_weight_initilization = params['weight_initilization']
        else:
            params_weight_initilization = None
        if "bias_initilization" in params:
            params_bias_initilization = params['bias_initilization']
        else:
            params_bias_initilization = None

        tmp = [self.in_dim] + [self.hid_dim] * self.num_hid_layer + [self.out_dim]
        self.mlp = torch.nn.ModuleList()
        self.weight_layer_indices = []
        with torch.no_grad():
            for i in range(len(tmp) - 2):
                lin = torch.nn.Linear(tmp[i], tmp[i + 1])
                self.layer_weight_initilizer(lin, params_weight_initilization)
                self.layer_bias_initilizer(lin, params_bias_initilization)
                self.mlp.append(lin)
                self.weight_layer_indices.append(i * 2)
                self.mlp.append(self.actFun)
        lin = torch.nn.Linear(tmp[-2], tmp[-1])
        self.layer_weight_initilizer(lin, params_weight_initilization)
        self.layer_bias_initilizer(lin, params_bias_initilization)
        self.mlp.append(lin)
        self.weight_layer_indices.append(i * 2+2)
        print("model initilized!")

    def load_parameters(self, nn_weights_path=str()) -> None:
        assert os.path.exists(nn_weights_path), f"{nn_weights_path} doesnt exist"
        self.load_state_dict(torch.load(nn_weights_path))
        print("model loaded!")

    @ staticmethod
    def layer_weight_initilizer(
        linear_lyer:torch.nn.Linear, 
        params: Optional[Dict] = None,
    )-> None:
        with torch.no_grad():
            if params is None or params['type'] == "gloret":
                std = np.sqrt(2.0/(sum(linear_lyer.weight.data.shape)))
                torch.nn.init.normal_(linear_lyer.weight, mean=0, std=std)
            elif params['type'] == "standard_guassian":
                std = params['std']
                torch.nn.init.normal_(linear_lyer.weight, mean=0, std=std)
            else:
                raise NotImplementedError(params['type'])

    @ staticmethod
    def layer_bias_initilizer(
        linear_lyer:torch.nn.Linear, 
        params: Optional[Dict] = None,
    )-> None:
        with torch.no_grad():
            if params is None:
                torch.nn.init.constant_(linear_lyer.bias, 0.)
            elif params['type'] == "constant":
                torch.nn.init.constant_(linear_lyer.bias, params['value'])
            else:
                raise NotImplementedError(params['type'])

    def get_network_architecture(self) -> List[int]:
        out = []
        for i in self.weight_layer_indices:
            out.append(self.mlp[i].weight.shape[1])
        out.append(self.mlp[i].weight.shape[0])
        return out


    def set_weights(self, np_seed:int=None, method=str()):
        """
            the method is not consistent with the litrature
            it is used for checking with tf implemetation
        """
        if np_seed is not None:
            np.random.seed(np_seed)
        
        for i in self.weight_layer_indices:
            layer = self.mlp[i]
            with torch.no_grad():
                w, b = layer.weight.data, layer.bias.data
                ww = np.random.randn(w.shape[0], w.shape[1]) / np.sqrt(max(w.shape))
                bb = np.random.randn(b.shape[0]) / np.sqrt(b.shape[0])
                
                layer.weight.data = torch.FloatTensor(ww)
                layer.bias.data =  torch.FloatTensor(bb)


    def forward(self, x: torch.tensor) -> torch.tensor:
        y = x
        for f in self.mlp:
            y = f(y)
        return y

    def get_total_number_params(self)->int:
        pp=0
        for p in list(self.parameters()):
            nn=1
            for s in list(p.size()):
                nn = nn*s
            pp += nn
        return pp

    def save(self, dir_to_save: str, model_info_name: str, weight_name: str) -> None:
        weight_path = os.path.join(dir_to_save, weight_name)
        torch.save(self.state_dict(), weight_path)
        tmp = {"weight_path": weight_path}
        for k, v in self.__dict__.items():
            if k in {"in_dim", "out_dim", "hid_dim", "num_hid_layer", "act_type"}:
                tmp[k] = v
        tmp["torch_version"] = torch.__version__
        tmp["model"] = "MLP"
        tmp_name_ = drop_file_type(model_info_name, "json")
        with open(os.path.join(dir_to_save, tmp_name_ + ".json"), "w") as f:
            json.dump(tmp, f)
        print("model saved!")
