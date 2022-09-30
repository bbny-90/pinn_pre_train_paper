from typing import Dict, Optional
import numpy as np
import torch
from models.nueral_net_pt import MLP
from pdes.operators import get_laplace_scalar
from trainer.helper import AuxilaryTaskScheduler
# from trainer.gradient_surgery import PCGrad

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
    base_optimizer = torch.optim.Adam(solution.parameters(), lr=lr)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau( 
            base_optimizer, mode='min', patience=lr_patience, factor=lr_red_factoe,
            min_lr=min_lr, verbose=False
    )
    #
    # optimizer = PCGrad(base_optimizer, lr_scheduler=lr_scheduler)

    
    u_bc_pt = torch.from_numpy(u_dbc).float().to(device=device)
    source_pde_pt = torch.from_numpy(source_pde).float().to(device=device)
    #
    loss_rec = {'bc':[], 'pde':[], 'val':[], 'lr':[]}
    loss_weights = {'bc':weight_pde, 'pde':weight_bc}
    tmp = sum(loss_weights.values())
    loss_weights = {k:v/tmp for k, v in loss_weights.items()}
    epsilon = 2.
    for epoch in range(epochs):
        # optimizer.zero_grad()
        base_optimizer.zero_grad()
        x_pde_pt = torch.from_numpy(x_pde).float().requires_grad_(True).to(device=device)
        x_dbc_pt = torch.from_numpy(x_dbc).float().requires_grad_(True).to(device=device)
        u_pde_pred = solution(x_pde_pt)
        u_dbc_pred = solution(x_dbc_pt)
        lap_u_pde = get_laplace_scalar(u_pde_pred,x_pde_pt, device)
        pde_res = (lap_u_pde - source_pde_pt).pow(2).mean()
        bc_res = (u_dbc_pred - u_bc_pt).pow(2).mean()

        loss_tot = loss_weights['pde'] * pde_res + loss_weights['bc'] * bc_res
        loss_tot.backward()
        base_optimizer.step()
        loss_tot = loss_tot.item()
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
        print(f'epoch {epoch}:', f'loss_tot {loss_tot:.8f} ', 
              f'pde_res {pde_res.item():.8f} bc_res {bc_res.item():.8f} val {val:0.8f}',
              f"w_bc {loss_weights['bc']} w_pde {loss_weights['pde']}"
        )
        loss_rec['pde'].append(pde_res.item())
        loss_rec['bc'].append(bc_res.item())
        loss_rec['val'].append(val.item())
        # loss_rec['lr'].append(optimizer._optim.param_groups[0]['lr'])
        loss_rec['lr'].append(base_optimizer.param_groups[0]['lr'])
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
    base_optimizer = torch.optim.Adam(solution.parameters(), lr=lr)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau( 
            base_optimizer, mode='min', patience=lr_patience, factor=lr_red_factoe,
            min_lr=min_lr, verbose=False
    )
    #
    # optimizer = PCGrad(base_optimizer, lr_scheduler=lr_scheduler)
    
    u_bc_pt = torch.from_numpy(u_dbc).float().to(device=device)
    source_pde_pt = torch.from_numpy(source_pde).float().to(device=device)
    u_guide_pt = torch.from_numpy(u_guide).float().to(device=device)
    #
    loss_rec = {'bc':[], 'pde':[], 'guide':[], 'val':[], 'lr':[]}
    loss_weights = {'bc':weight_pde, 'pde':weight_bc, 'guide':weight_guide}
    aux_scheduler = AuxilaryTaskScheduler(auxilary_task_params)
    for epoch in range(epochs):
        loss_weights['guide'] = aux_scheduler(curent_penalty=loss_weights['guide'], epoch=epoch)
        tmp = sum(loss_weights.values())
        loss_weights = {k:v/tmp for k, v in loss_weights.items()}
        del tmp
        # optimizer.zero_grad()
        base_optimizer.zero_grad()
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

        loss_tot = pde_res * loss_weights['pde'] +\
            bc_res * loss_weights['bc']+\
            guide_res * loss_weights['guide']      
        loss_tot.backward()
        base_optimizer.step()
        loss_tot = loss_tot.item()
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
        print(f'epoch {epoch}:', f'loss_tot {loss_tot:.6f} ', 
              f'pde_res {pde_res.item():.6f} bc_res {bc_res.item():.6f}', 
              f'guide_res {guide_res.item():0.6f} val {val:0.6f}',
              f'w_pde {loss_weights["pde"]:0.3f}',
              f'w_bc {loss_weights["bc"]:0.3f}',
              f'w_guide {loss_weights["guide"]:0.3f}',
        )
        loss_rec['pde'].append(pde_res.item())
        loss_rec['bc'].append(bc_res.item())
        loss_rec['guide'].append(guide_res.item())
        loss_rec['val'].append(val.item())
        # loss_rec['lr'].append(optimizer._optim.param_groups[0]['lr'])
        loss_rec['lr'].append(base_optimizer.param_groups[0]['lr'])
    return loss_rec