import copy
import random
from typing import List, Optional, Tuple
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np

class PCGrad():
    def __init__(self, 
            optimizer:Optimizer, 
            lr_scheduler:Optional[ReduceLROnPlateau]=None,
        ):
        """
            this is a multi-objective optimizer that aims to suppress the gradient conflicts
            based on the following paper
            Gradient Surgery for Multi-Task Learning
            T. Yu, et al 2020

        """
        self._optim = optimizer
        self._lr_scheduler = lr_scheduler

    @property
    def optimizer(self):
        return self._optim

    def step_lr_scheduler(self, val):
        self._lr_scheduler.step(val)

    def zero_grad(self):
        return self._optim.zero_grad(set_to_none=True)

    def step(self):
        return self._optim.step()

    def backward_regular(self, 
        objectives:List[torch.Tensor], 
        objectives_weights: List[float]
        )-> None:
        assert len(objectives) == len(objectives_weights),\
            (len(objectives) == len(objectives_weights))
        loss = torch.tensor(0.).to(objectives[0].device)
        for obj, w in zip(objectives, objectives_weights):
            loss += (obj.mul(w))
        loss.backward()


    def backward_surgery(self, objectives:List[torch.Tensor]):
        grads, shapes = self._get_grad_params_wrt_all_objective(objectives)
        pc_grad = self._project_conflict(grads)
        pc_grad = self._unflatten_grad(pc_grad, shapes[0])
        self._set_grad(pc_grad)


    def _project_conflict(self, 
            grads:List[torch.Tensor],
        )->torch.Tensor:
        """
            here we implement alog 1 in page 4 of the following paper
            https://arxiv.org/pdf/2001.06782.pdf
        """
        pc_grad = copy.deepcopy(grads)
        for g_i in pc_grad:
            random.shuffle(grads)
            for g_j in grads:
                g_i_g_j = torch.dot(g_i, g_j)
                if g_i_g_j < 0:
                    g_i -= (g_i_g_j) * g_j / (g_j.norm()**2) # TODO: avoid zero division
        return torch.stack(pc_grad).mean(dim=0).to(grads[0].device)

    def _set_grad(self, grads):
        '''
            set the modified gradients to the network
        '''
        idx = 0
        for group in self._optim.param_groups:
            for p in group['params']:
                p.grad = grads[idx]
                idx += 1
        return

    def _get_grad_params_wrt_all_objective(self, 
            objectives: List[torch.Tensor]
        )->Tuple[List[torch.Tensor], List[Tuple[int, int]]]:
        grads, shapes = [], []
        for obj in objectives:
            self._optim.zero_grad(set_to_none=True)
            grad, shape = self._get_grad_params_wrt_objective(obj)
            flatten_grad = torch.cat([g.flatten() for g in grad])
            grads.append(flatten_grad)
            shapes.append(shape)
        return grads, shapes

    def _unflatten_grad(self, grads, shapes):
        unflatten_grad, idx = [], 0
        for shape in shapes:
            length = np.prod(shape)
            unflatten_grad.append(grads[idx:idx + length].view(shape).clone())
            idx += length
        return unflatten_grad


    def _get_grad_params_wrt_objective(self, obj:torch.Tensor
        )-> Tuple[List[torch.Tensor], List[Tuple[int, int]]]:
        '''
            get the gradient of the parameters of the network wrt a specific objective
            
            output:
            - grad: gradient of the parameters
            - shape: shape of the parameters
            - trainable: whether the parameter is trainable
        '''
        assert obj.dim() == 0, obj.dim()
        obj.backward(retain_graph=True)
        grad, shape = [], []
        for group in self._optim.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    grad.append(p.grad.clone())
                    shape.append(p.grad.shape)
                else:
                    # This could happend for some uncommon loss functions
                    # e.g., when applying gradient operator, a parametr contribution 
                    # (e.g., the bias term in the last layer of MLP for a PINN problem)
                    # can be gone so there is no back propagation
                    #
                    # NOTE: this break differentiability for possible downstream tasks
                    grad.append(torch.zeros_like(p))
                    shape.append(p.shape)
        return grad, shape