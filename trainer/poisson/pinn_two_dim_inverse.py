from typing import Dict, Optional
import numpy as np
import torch
from models.nueral_net_pt import MLP
from pdes.operators import get_divergence_vector
from trainer.gradient_surgery import PCGrad

def train(
    solution:MLP,
    x_pde: np.ndarray, source_pde:np.ndarray,
    x_dbc:np.ndarray, u_dbc:np.ndarray,
    x_guide:np.ndarray, u_guide:np.ndarray, flux_guide:np.ndarray,
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
    need_surgury = train_params['need_surgury']
    weight_pde = train_params['weight_pde']
    weight_dbc = train_params['weight_dbc']
    weight_guide_u = train_params['weight_guide_u']
    weight_guide_q0 = train_params['weight_guide_flux0']
    weight_guide_q1 = train_params['weight_guide_flux1']
    weight_pos_def_perm = train_params['weight_pos_def_perm']
    tol_pos_def = train_params['tol_pos_def']
    #
    base_optimizer = torch.optim.Adam(
        [
            {'params':solution.mlp.parameters()},
            {'params':[solution._perm_params]}
        ], lr=lr
    )

    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau( 
            base_optimizer, mode='min', patience=lr_patience, factor=lr_red_factoe,
            min_lr=min_lr, verbose=False
    )
    #
    optimizer = PCGrad(base_optimizer, lr_scheduler=lr_scheduler)
    
    x_dbc_pt = torch.from_numpy(x_dbc).float().requires_grad_(True).to(device=device).requires_grad_(False)
    u_dbc_pt = torch.from_numpy(u_dbc).float().to(device=device).requires_grad_(False)
    
    x_pde_pt = torch.from_numpy(x_pde).float().requires_grad_(True).to(device=device).requires_grad_(True)
    source_pde_pt = torch.from_numpy(source_pde).float().to(device=device).requires_grad_(False)
    
    x_guide_pt = torch.from_numpy(x_guide).float().to(device=device).requires_grad_(True)
    u_guide_pt = torch.from_numpy(u_guide).float().to(device=device).requires_grad_(False)
    flux_guide_pt = torch.from_numpy(flux_guide).float().to(device=device).requires_grad_(False)

    solution._perm_params.to(device=device)
    
    
    #
    loss_rec = {'dbc':[], 'pde':[], 'guide_u':[], 
                'guide_q0':[], 'guide_q1':[], 
                'pos_def_perm':[],
                'val':[], 'lr':[]}
    weights_init = {'dbc':weight_dbc, 'pde':weight_pde, 'guide_u':weight_guide_u, 
                'guide_q0':weight_guide_q0, 'guide_q1':weight_guide_q1, 
                'pos_def_perm':weight_pos_def_perm,}
    tmp = sum(weights_init.values())
    weights_init = {k:v/tmp for k, v in weights_init.items()}
    discovered_perm = []
    for epoch in range(epochs):
        # optimizer.zero_grad()
        base_optimizer.zero_grad()
        perm = solution.get_perm()
        u_pde_pred = solution(x_pde_pt)
        u_dbc_pred = solution(x_dbc_pt)
        u_guide_pred = solution(x_guide_pt)
        # pde
        flux_guide_pred = solution.calc_flux(u=u_pde_pred, x=x_pde_pt, perm=perm, device=device)
        pde_res = (get_divergence_vector(flux_guide_pred, x_pde_pt) - source_pde_pt).pow(2).mean() * weights_init['pde']

        # dbc
        dbc_res = (u_dbc_pred - u_dbc_pt).pow(2).mean()
        # exp u
        guide_u_res = (u_guide_pred - u_guide_pt).pow(2).mean() * weights_init['guide_u']
        # exp flux
        flux_guide_pred = solution.calc_flux(u=u_guide_pred, x=x_guide_pt, perm=perm, device=device)
        guide_q0_res = (flux_guide_pred[:, 0] - flux_guide_pt[:, 0]).pow(2).mean() * weights_init['guide_q0']
        guide_q1_res = (flux_guide_pred[:, 1] - flux_guide_pt[:, 1]).pow(2).mean() * weights_init['guide_q1']
        # pos def perm
        pos_def_perm_res = torch.relu(- torch.det(perm) + tol_pos_def) * weights_init['pos_def_perm']
        #
        objectives = [pde_res, dbc_res, guide_u_res, guide_q0_res, guide_q1_res, pos_def_perm_res]
        if need_surgury:
            optimizer.backward_surgery(objectives)
        else:
            optimizer.backward_regular(objectives)
        optimizer.step()
        with torch.no_grad():
            loss_tot = sum(objectives).item()
        if epoch > lr_sch_epoch:
            optimizer.step_lr_scheduler(loss_tot)
        if x_val is not None:
            with torch.no_grad():
                u_val_pred = solution(torch.from_numpy(x_val).float().to(device=device))
                val = np.mean(
                    np.sum((u_val_pred.detach().numpy() - u_val)**2, axis=1)
                )
        else:
            val = -1.
        print(f'epoch {epoch}:', f'loss_tot {loss_tot:.6f} ', 
              f'pde_res {pde_res.item():.6f} dbc_res {dbc_res.item():.6f}', 
              f'guide_u_res {guide_u_res.item():0.6f}',
              f'guide_q0_res {guide_q0_res.item():0.6f}',
              f'guide_q1_res {guide_q0_res.item():0.6f}',
              f'val {val:0.6f}',
              f'w_pde {weights_init["pde"]:0.3f}',
              f'w_dbc {weights_init["dbc"]:0.3f}',
              f'w_guide_u {weights_init["guide_u"]:0.3f}',
              f'w_guide_q0 {weights_init["guide_q0"]:0.3f}',
              f'w_guide_q1 {weights_init["guide_q1"]:0.3f}',
        )
        loss_rec['pde'].append(pde_res.item())
        loss_rec['dbc'].append(dbc_res.item())
        loss_rec['guide_u'].append(guide_u_res.item())
        loss_rec['guide_q0'].append(guide_q0_res.item())
        loss_rec['guide_q1'].append(guide_q1_res.item())
        loss_rec['pos_def_perm'].append(pos_def_perm_res.item())
        loss_rec['val'].append(val)
        loss_rec['lr'].append(optimizer._optim.param_groups[0]['lr'])
        with torch.no_grad():
            discovered_perm.append(
                perm.detach().numpy().flatten()
            )
    discovered_perm = np.stack(discovered_perm, axis=0)
    return loss_rec, discovered_perm