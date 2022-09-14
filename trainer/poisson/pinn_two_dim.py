from typing import Dict, Optional
import numpy as np
import torch
from models.nueral_net_pt import MLP
from pdes.operators import get_laplace_scalar
from trainer.helper import AuxilaryTaskScheduler

def train_vanilla(
    solution:MLP,
    x_pde: np.ndarray, source_pde:np.ndarray,
    x_dbc:np.ndarray, u_dbc:np.ndarray,
    loss_weights:Dict[str, float],
    train_params: dict,
    device: torch.device,
    x_val:Optional[np.ndarray] = None, 
    u_val:Optional[np.ndarray] = None,
)-> Dict[str, list]:
    lr = train_params['lr']
    min_lr = train_params['min_lr']
    lr_patience = train_params['lr_patience']
    lr_red_factoe = train_params['lr_red_factoe']
    epochs = train_params['epochs']
    lr_sch_epoch = train_params['lr_sch_epoch']
    weight_pde = loss_weights['pde']
    weight_bc = loss_weights['bc']
    #
    optimizer = torch.optim.Adam(solution.parameters(), lr=lr)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau( 
            optimizer, mode='min', patience=lr_patience, factor=lr_red_factoe,
            min_lr=min_lr, verbose=False
    )
    #

    
    u_bc_pt = torch.from_numpy(u_dbc).float().to(device=device)
    source_pde_pt = torch.from_numpy(source_pde).float().to(device=device)
    #
    loss_rec = {'bc':[], 'pde':[], 'val':[], 'lr':[]}
    for epoch in range(epochs):
        optimizer.zero_grad()
        x_pde_pt = torch.from_numpy(x_pde).float().requires_grad_(True).to(device=device)
        x_dbc_pt = torch.from_numpy(x_dbc).float().requires_grad_(True).to(device=device)
        u_pde_pred = solution(x_pde_pt)
        u_dbc_pred = solution(x_dbc_pt)
        lap_u_pde = get_laplace_scalar(u_pde_pred,x_pde_pt, device)
        pde_res = (lap_u_pde - source_pde_pt).pow(2).mean()
        bc_res = (u_dbc_pred - u_bc_pt).pow(2).mean()
        loss_tot = weight_pde * pde_res + weight_bc * bc_res
        loss_tot.backward()
        optimizer.step()
        if epoch > lr_sch_epoch:
            lr_scheduler.step(loss_tot)
        if x_val is not None:
            with torch.no_grad():
                u_val_pred = solution(torch.from_numpy(x_val).float().to(device=device))
                val = np.mean(
                    np.sum((u_val_pred.detach().numpy() - u_val)**2, axis=1)
                )
        else:
            val = -1.
        print(f'epoch {epoch}:', f'loss_tot {loss_tot.item():.8f} ', 
              f'pde_res {pde_res.item():.8f} bc_res {bc_res.item():.8f} val {val:0.8f}'
        )
        loss_rec['pde'].append(pde_res.item())
        loss_rec['bc'].append(bc_res.item())
        loss_rec['val'].append(val.item())
        loss_rec['lr'].append(optimizer.param_groups[0]['lr'])
    return loss_rec


def train_guided(
    solution:MLP,
    x_pde: np.ndarray, source_pde:np.ndarray,
    x_dbc:np.ndarray, u_dbc:np.ndarray,
    x_guide:np.ndarray, u_guide:np.ndarray,
    loss_weights:Dict[str, float],
    train_params: dict,
    device: torch.device,
    x_val:Optional[np.ndarray] = None, 
    u_val:Optional[np.ndarray] = None,
)-> Dict[str, list]:
    lr = train_params['lr']
    min_lr = train_params['min_lr']
    lr_patience = train_params['lr_patience']
    lr_red_factoe = train_params['lr_red_factoe']
    epochs = train_params['epochs']
    lr_sch_epoch = train_params['lr_sch_epoch']
    auxilary_task_params:dict = train_params['auxilary_task_params']
    weight_pde = loss_weights['pde']
    weight_bc = loss_weights['bc']
    weight_guide = loss_weights['guide']
    #
    optimizer = torch.optim.Adam(solution.parameters(), lr=lr)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau( 
            optimizer, mode='min', patience=lr_patience, factor=lr_red_factoe,
            min_lr=min_lr, verbose=False
    )
    #

    
    u_bc_pt = torch.from_numpy(u_dbc).float().to(device=device)
    source_pde_pt = torch.from_numpy(source_pde).float().to(device=device)
    u_guide_pt = torch.from_numpy(u_guide).float().to(device=device)
    #
    loss_rec = {'bc':[], 'pde':[], 'val':[], 'lr':[]}
    aux_scheduler = AuxilaryTaskScheduler(auxilary_task_params)
    for epoch in range(epochs):
        weight_guide = aux_scheduler(curent_penalty=weight_guide, epoch=epoch)
        optimizer.zero_grad()
        x_pde_pt = torch.from_numpy(x_pde).float().requires_grad_(True).to(device=device)
        x_dbc_pt = torch.from_numpy(x_dbc).float().requires_grad_(True).to(device=device)
        x_guide_pt = torch.from_numpy(x_guide).float().to(device=device)
        u_pde_pred = solution(x_pde_pt)
        u_dbc_pred = solution(x_dbc_pt)
        u_guide_pred = solution(x_guide_pt)
        lap_u_pde = get_laplace_scalar(u_pde_pred,x_pde_pt, device)
        pde_res = (lap_u_pde - source_pde_pt).pow(2).mean()
        bc_res = (u_dbc_pred - u_bc_pt).pow(2).mean()
        guide_res = (u_guide_pred - u_guide_pt).pow(2).mean()
        loss_tot = weight_pde * pde_res + weight_bc * bc_res + weight_guide * guide_res
        loss_tot.backward()
        optimizer.step()
        if epoch > lr_sch_epoch:
            lr_scheduler.step(loss_tot)
        if x_val is not None:
            with torch.no_grad():
                u_val_pred = solution(torch.from_numpy(x_val).float().to(device=device))
                val = np.mean(
                    np.sum((u_val_pred.detach().numpy() - u_val)**2, axis=1)
                )
        else:
            val = -1.
        print(f'epoch {epoch}:', f'loss_tot {loss_tot.item():.8f} ', 
              f'pde_res {pde_res.item():.8f} bc_res {bc_res.item():.8f}', 
              f'guide_res {guide_res.item():0.8f} val {val:0.8f}'
        )
        loss_rec['pde'].append(pde_res.item())
        loss_rec['bc'].append(bc_res.item())
        loss_rec['val'].append(val.item())
        loss_rec['lr'].append(optimizer.param_groups[0]['lr'])
    return loss_rec