from __future__ import annotations
from turtle import pd
from typing import Dict, List, Tuple
import os
import json
import torch
import numpy as np

from helper.other import drop_file_type

def get_remap_array(
    hold:int, hnew:int
    )-> Tuple[List[int], List[int]]:
    """
        This function is used for net2net widening operation
        see page 4: https://arxiv.org/pdf/1511.05641.pdf
        ---------
        input
            hold: num of hidden units for a layer before widening
            hnew: num of hidden units for a layer after widening
        ---------
        output
           remap_arr: an array that indicates the association of the new state 
                      with the old hidden state
           counts: an array that shows the number of duplication of the 
                    old indices in the new satte
    """
    assert hold <= hnew, (hold, hnew)
    remap_arr = np.arange(hnew)
    remap_arr[hold:] = np.random.choice(hold, hnew - hold)
    counts = np.zeros(hold, dtype=int)
    for i in remap_arr:
        counts[i] += 1
    return remap_arr, counts

def get_remap_layers(
    arct_old:List[int],arct_new:List[int]
    )-> Dict[int, Dict[str, List[int]]]:
    """
        arct_old: MLP architecture befor widening (input and output layers included)
        arct_new: MLP architecture after widening (input and output layers included)
                this includes input and output layers as well!
    """
    assert len(arct_old) == len(arct_new), (arct_old, arct_new)
    remap_all = {}
    i = 0
    for hold, hnew in zip(arct_old, arct_new):
        remap_arr, counts = get_remap_array(hold, hnew)
        remap_all[i] = {"remap_arr":remap_arr, "counts":counts}
        i += 1
    return remap_all


class MLP(torch.nn.Module):
    def __init__(self, params: dict, nn_weights_path=str()) -> None:
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
        else:
            raise NotImplementedError(f"activation {self.act_type} is not supported")

        tmp = [self.in_dim] + [self.hid_dim] * self.num_hid_layer + [self.out_dim]
        self.mlp = torch.nn.ModuleList()
        self.weight_layer_indices = []
        with torch.no_grad():
            for i in range(len(tmp) - 2):
                lin = torch.nn.Linear(tmp[i], tmp[i + 1])
                self.layer_weight_initilizer(lin)
                self.mlp.append(lin)
                self.weight_layer_indices.append(i * 2)
                self.mlp.append(self.actFun)
        lin = torch.nn.Linear(tmp[-2], tmp[-1])
        self.layer_weight_initilizer(lin)
        self.mlp.append(lin)
        self.weight_layer_indices.append(i * 2+2)
        if nn_weights_path:
            assert os.path.exists(nn_weights_path), f"{nn_weights_path} doesnt exist"
            self.load_state_dict(torch.load(nn_weights_path))
            print("model loaded!")
        else:
            print("model initilized!")

    @ staticmethod
    def layer_weight_initilizer(
        linear_lyer:torch.nn.Linear, init_type = "gloret",
    )-> None:
        if init_type == "gloret":
            with torch.no_grad():
                torch.nn.init.constant_(linear_lyer.bias, 0.)
                std = np.sqrt(2.0/(sum(linear_lyer.weight.data.shape)))
                torch.nn.init.normal_(linear_lyer.weight, mean=0, std=std)
        else:
            raise NotImplementedError()

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

    def widen(self, new_hid_dim:int) -> MLP:
        """
            input and output layers remain the same size
        """
        # TODO: make sure the other places can see these changes
        current_arch = self.get_network_architecture()
        in_dim, out_dim = current_arch[0], current_arch[-1]
        current_hid_dim = current_arch[1]
        assert current_hid_dim <= new_hid_dim, (current_hid_dim, new_hid_dim)
        params_new = {
            "in_dim":in_dim, "out_dim":out_dim,
            "hid_dim":new_hid_dim,
            "num_hid_layer":len(current_arch) - 2,
            "act_type":self.act_type
        }
        widen_net = MLP(params=params_new)
        remap_all = get_remap_layers(
            current_arch, widen_net.get_network_architecture()
        )
        # print(widen_net.weight_layer_indices)
        # print(widen_net.get_network_architecture())
        # for jj in widen_net.mlp:
        #     print(jj.weight.data.shape)
        # exit()
        # here we implement exactly the equation at the middle of page 5
        # https://arxiv.org/pdf/1511.05641.pdf
        with torch.no_grad():
            for i, layIndx in enumerate(widen_net.weight_layer_indices):
                W = widen_net.mlp[layIndx].weight.data
                g = remap_all[i]['remap_arr']
                gg = remap_all[i+1]['remap_arr']
                counts = remap_all[i]['counts']
                for k in range(W.shape[1]):
                    for j in range(W.shape[0]):
                        widen_net.mlp[layIndx].weight.data[j, k] =\
                            self.mlp[layIndx].weight.data[gg[j], g[k]] / counts[g[k]]

                g = remap_all[i+1]['remap_arr']
                counts = remap_all[i+1]['counts']
                for k in range(W.shape[0]):
                    widen_net.mlp[layIndx].bias.data[k] =\
                        self.mlp[layIndx].bias.data[g[k]]
        return widen_net

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
