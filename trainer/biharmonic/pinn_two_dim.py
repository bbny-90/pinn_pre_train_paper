from typing import Dict, Optional
import numpy as np
import torch
from models.nueral_net_pt import MLP
from pdes.operators import get_divergence_tensor, get_gradient_vector
from trainer.helper import AuxilaryTaskScheduler
from mechnics.solid import ElasticityLinear

def train_vanilla_mixed_disp_strain(
    solution:MLP,
    mat_model:ElasticityLinear,
    x_pde: np.ndarray, 
    source_pde:np.ndarray,
    #
    x_dbc:np.ndarray, 
    u_dbc:np.ndarray,
    #
    x_nbc:np.ndarray, 
    n_nbc:np.ndarray, 
    trac_nbc:np.ndarray,
    #
    loss_weights:Dict[str, float],
    train_params: dict,
    device: torch.device,
    #
    x_val:Optional[np.ndarray] = None, 
    source_pde_val:Optional[np.ndarray] = None,
)-> Dict[str, list]:
    
    lr = train_params['lr']
    min_lr = train_params['min_lr']
    lr_patience = train_params['lr_patience']
    lr_red_factoe = train_params['lr_red_factoe']
    epochs = train_params['epochs']
    lr_sch_epoch = train_params['lr_sch_epoch']
    weight_pde = loss_weights['pde']
    weight_dbc = loss_weights['dbc']
    weight_nbc = loss_weights['nbc']
    weight_compat = loss_weights['compat']
    #
    optimizer = torch.optim.Adam(solution.parameters(), lr=lr)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau( 
            optimizer, mode='min', patience=lr_patience, factor=lr_red_factoe,
            min_lr=min_lr, verbose=False
    )
    #
    u_dbc_pt = torch.from_numpy(u_dbc).float().to(device=device)
    x_nbc_pt = torch.from_numpy(x_nbc).float().requires_grad_(True).to(device=device)
    n_nbc_pt = torch.from_numpy(n_nbc).float().requires_grad_(True).to(device=device)
    trac_nbc_pt = torch.from_numpy(trac_nbc).float().requires_grad_(True).to(device=device)
    x_dbc_pt = torch.from_numpy(x_dbc).float().requires_grad_(True).to(device=device)
    source_pde_pt = torch.from_numpy(source_pde).float().to(device=device)
    if source_pde_val is not None:
        source_pde_val_pt = torch.from_numpy(source_pde_val).float().to(device=device)
    #
    loss_rec = {'dbc':[], 'nbc':[], 'pde':[], 'compat':[], 
                'pde_val':[], 'compat_val':[], 'lr':[]
    }
    for epoch in range(epochs):
        optimizer.zero_grad()
        x_pde_pt = torch.from_numpy(x_pde).float().requires_grad_(True).to(device=device)
        temp_pde_pred = solution(x_pde_pt)
        u_pde_pred_vec = temp_pde_pred[:, :2]
        eps_pde_pred_vec = temp_pde_pred[:, 2:]
        sig_pde_pred_vec = mat_model.get_stress(eps_pde_pred_vec)
        sig_pde_pred_tens = mat_model.get_stress_tensor_from_stress_vector(sig_pde_pred_vec)
        div_sig = get_divergence_tensor(sig_pde_pred_tens, x_pde_pt, device)
        # pde residual
        pde_res = (div_sig - source_pde_pt).pow(2).sum(dim=1).mean()
        eps_from_u_pde = get_gradient_vector(u_pde_pred_vec, x_pde_pt)
        # compatibility condition
        compat_res = (
            eps_from_u_pde - mat_model.get_strain_tensor_from_strain_vector(eps_pde_pred_vec)
        ).pow(2).sum([1, 2]).mean()
        # dirichlet bc
        u_dbc_pred = solution(x_dbc_pt)[:, :2]
        dbc_res = (u_dbc_pred - u_dbc_pt).pow(2).sum(dim=1).mean()

        eps_nbc_vec = solution(x_nbc_pt)[:, 2:]
        sig_nbc_vec = mat_model.get_stress(eps_nbc_vec)
        sig_nbc_tens = mat_model.get_stress_tensor_from_stress_vector(sig_nbc_vec)
        normal_trac_nbc = torch.einsum("nij,nj->ni", sig_nbc_tens, n_nbc_pt)
        # nuemann bc
        nbc_res = (normal_trac_nbc - trac_nbc_pt).pow(2).sum(dim=1).mean()
        # total loss
        loss_tot = weight_pde * pde_res + weight_dbc * dbc_res + weight_nbc * nbc_res + weight_compat * compat_res
        loss_tot.backward()
        optimizer.step()
        if epoch > lr_sch_epoch:
            lr_scheduler.step(loss_tot)
        if x_val is not None:
            x_val_pt = torch.from_numpy(x_val).float().requires_grad_(True).to(device=device)
            temp_val = solution(x_val_pt)
            u_val = temp_val[:, :2]
            eps_val_vec = temp_val[:, 2:]
            sig_val_vec = mat_model.get_stress(eps_val_vec)
            sig_val_tens = mat_model.get_stress_tensor_from_stress_vector(sig_val_vec)
            div_sig = get_divergence_tensor(sig_val_tens, x_val_pt, device)
            pde_res_val = (div_sig - source_pde_val_pt).pow(2).sum(dim=1).mean().item()

            eps_from_u_pde = get_gradient_vector(u_val, x_val_pt)
            compat_res_val = (
                eps_from_u_pde - mat_model.get_strain_tensor_from_strain_vector(eps_val_vec)
            ).pow(2).sum([1, 2]).mean().item()
        else:
            pde_res_val = -1.
            compat_res_val = -1.
        print(f'epoch {epoch}:', f'loss_tot {loss_tot.item():.8f} ', 
              f'pde_res {pde_res.item():.8f} dbc_res {dbc_res.item():.8f} ',
              f'nbc_res {nbc_res.item():.8f} compt_res {compat_res.item():.8f} ', 
              f'pde_res_val {pde_res_val:0.8f} compt_res_val {compat_res_val:0.8f}'
        )
        loss_rec['pde'].append(pde_res.item())
        loss_rec['dbc'].append(dbc_res.item())
        loss_rec['nbc'].append(nbc_res.item())
        loss_rec['compat'].append(compat_res.item())
        loss_rec['pde_val'].append(pde_res_val)
        loss_rec['compat_val'].append(compat_res_val)
        loss_rec['lr'].append(optimizer.param_groups[0]['lr'])
    return loss_rec