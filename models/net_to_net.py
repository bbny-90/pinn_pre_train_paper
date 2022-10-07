from typing import (
    List,
    Tuple,
    Dict
)
import numpy as np
import torch
from models.nueral_net_pt import MLP

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


def widen_net_to_net(old_net:MLP, new_hid_dim:int, **others) -> MLP:
    """
        input and output layers remain the same size
    """
    # TODO: make sure the other places can see these changes
    current_arch = old_net.get_network_architecture()
    in_dim, out_dim = current_arch[0], current_arch[-1]
    current_hid_dim = current_arch[1]
    assert current_hid_dim <= new_hid_dim, (current_hid_dim, new_hid_dim)
    params_new = {
        "in_dim":in_dim, "out_dim":out_dim,
        "hid_dim":new_hid_dim,
        "num_hid_layer":len(current_arch) - 2,
        "act_type":old_net.act_type
    }
    others['mlp_config'] = params_new
    widen_net = old_net.__class__(**others)
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
                        old_net.mlp[layIndx].weight.data[gg[j], g[k]] / counts[g[k]]

            g = remap_all[i+1]['remap_arr']
            counts = remap_all[i+1]['counts']
            for k in range(W.shape[0]):
                widen_net.mlp[layIndx].bias.data[k] =\
                    old_net.mlp[layIndx].bias.data[g[k]]
    return widen_net

