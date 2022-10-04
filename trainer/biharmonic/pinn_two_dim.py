from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from models.nueral_net_pt import MLP
from pdes.operators import get_divergence_tensor, get_gradient_vector
from trainer.helper import AuxilaryTaskScheduler
from mechnics.solid import ElasticityLinear
from trainer.gradient_surgery import PCGrad
from helper.data_structures import (
    DirichletPoints,
    NeumannPoints,
    InternalPoints,
    InternalPointsSolidMixedForm
)

def transform_to_torch(
    pde_pnts: Optional[List[InternalPoints]],
    dbc_pnts: Optional[List[DirichletPoints]],
    nbc_pnts: Optional[List[NeumannPoints]],
    guide_pnts: Optional[List[InternalPointsSolidMixedForm]],
    device, 
    ) -> Tuple[List[InternalPoints], List[DirichletPoints], List[NeumannPoints], List[InternalPointsSolidMixedForm]]:
    to_torch = lambda x: torch.tensor(x).float().to(device)
    
    pde_pnt_pt:List[InternalPoints] = []
    for pde_pnt in pde_pnts:
        tmp = InternalPoints(
            to_torch(pde_pnt.x).requires_grad_(True),
            to_torch(pde_pnt.source).requires_grad_(False)
        )
        pde_pnt_pt.append(tmp)
    dbc_pnt_pt:Optional[List[DirichletPoints]] = []
    for dbc_pnt in dbc_pnts:
        tmp = DirichletPoints(
            to_torch(dbc_pnt.x).requires_grad_(False),
            to_torch(dbc_pnt.val).requires_grad_(False),
        )
        dbc_pnt_pt.append(tmp)
    nbc_pnt_pt: Optional[List[NeumannPoints]] = []
    for nbc_pnt in nbc_pnts:
        tmp = NeumannPoints(
            to_torch(nbc_pnt.x).requires_grad_(False),
            to_torch(nbc_pnt.val).requires_grad_(False),
            to_torch(nbc_pnt.normal).requires_grad_(False)
        )
        nbc_pnt_pt.append(tmp)
    guide_pnt_pt: Optional[List[InternalPointsSolidMixedForm]] = []
    for nbc_pnt in guide_pnts:
        tmp = InternalPointsSolidMixedForm(
            to_torch(nbc_pnt.x).requires_grad_(True),
            to_torch(nbc_pnt.source).requires_grad_(False),
            to_torch(nbc_pnt.disp).requires_grad_(False),
            to_torch(nbc_pnt.strain).requires_grad_(False)
        )
        guide_pnt_pt.append(tmp)
    return pde_pnt_pt, dbc_pnt_pt, nbc_pnt_pt, guide_pnt_pt

def get_normalization_factor(
    dbc_pnts: Optional[List[DirichletPoints]],
    guide_pnts: List[InternalPointsSolidMixedForm],
    ) -> Tuple:
    u = []
    eps = []
    for dbc_pnt in dbc_pnts: u.append(dbc_pnt.val)
    for nbc_pnt in guide_pnts:
        u.append(nbc_pnt.disp)
        eps.append(nbc_pnt.strain)
    std_u, std_eps = None, None
    if u:
        std_u = np.concatenate(u, axis=0).std(axis=0)
    if eps:
        std_eps = np.concatenate(eps, axis=0).std(axis=0)
    return std_u, std_eps
    


# TODO: (too many arguments) either the api or structure should be changed
def train_guided_mixed_disp_strain(
    *,
    solution:MLP,
    mat_model:ElasticityLinear,
    pde_pnts: Optional[List[InternalPoints]],
    dbc_pnts: Optional[List[DirichletPoints]],
    nbc_pnts: Optional[List[DirichletPoints]],
    # guide
    guide_pnts: List[InternalPointsSolidMixedForm],
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
    auxilary_task_params_u:dict = train_params['auxilary_task_params']['u']
    auxilary_task_params_eps:dict = train_params['auxilary_task_params']['eps']
    need_surgury = train_params['need_surgury']
    weight_pde = loss_weights['pde']
    weight_dbc = loss_weights['dbc']
    weight_nbc = loss_weights['nbc']
    weight_compat = loss_weights['compat']
    weight_guide_u = loss_weights['u_guide']
    weight_guide_eps = loss_weights['eps_guide']
    #
    base_optimizer = torch.optim.Adam(solution.parameters(), lr=lr)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau( 
            base_optimizer, mode='min', patience=lr_patience, factor=lr_red_factoe,
            min_lr=min_lr, verbose=False
    )
    optimizer = PCGrad(base_optimizer, lr_scheduler=lr_scheduler)
    #
    std_u, std_eps = get_normalization_factor(dbc_pnts, guide_pnts)
    std_eps = mat_model.get_strain_tensor_from_strain_vector(std_eps.reshape(1, -1)).squeeze()
    std_u = torch.tensor(std_u).float().to(device).requires_grad_(False)
    std_eps = torch.tensor(std_eps).float().to(device).requires_grad_(False)
    

    pde_pnts_pt, dbc_pnts_pt, nbc_pnts_pt, guide_pnts_pt =\
         transform_to_torch(pde_pnts, dbc_pnts, nbc_pnts, guide_pnts, device)
    
    
    loss_rec = {'dbc':[], 'nbc':[], 'pde':[], 'compat':[], 
                'u_guide':[], 'eps_guide':[],
                'pde_val':[], 'compat_val':[], 'lr':[]
    }
    calc_f_norm = lambda tens: tens.pow(2).sum([1, 2]).mean()
    aux_scheduler_u = AuxilaryTaskScheduler(auxilary_task_params_u)
    aux_scheduler_eps = AuxilaryTaskScheduler(auxilary_task_params_eps)
    for epoch in range(epochs):
        weight_guide_u = aux_scheduler_u(curent_penalty=weight_guide_u, epoch=epoch)
        weight_guide_eps = aux_scheduler_eps(curent_penalty=weight_guide_eps, epoch=epoch)
        optimizer.zero_grad()
        pde_res = torch.tensor(0.)
        compat_res = torch.tensor(0.)
        for pde_info in pde_pnts_pt:
            u_pde_pred_vec, eps_pde_pred_vec = solution(pde_info.x)
            # pde constraint
            sig_pde_pred_vec = mat_model.get_stress(eps_pde_pred_vec)
            sig_pde_pred_tens = mat_model.get_stress_tensor_from_stress_vector(sig_pde_pred_vec)
            div_sig = get_divergence_tensor(sig_pde_pred_tens, pde_info.x, device)
            pde_res += (div_sig - pde_info.source).pow(2).sum(dim=1).mean()
            # compatibility constraint
            eps_from_u_pde = get_gradient_vector(u_pde_pred_vec, pde_info.x)
            eps_pde_pred_tens = mat_model.get_strain_tensor_from_strain_vector(eps_pde_pred_vec)
            compat_res += calc_f_norm((eps_from_u_pde - eps_pde_pred_tens))
        del pde_info
        u_guide_res, eps_guide_res = torch.tensor(0.), torch.tensor(0.)
        for guide_info in guide_pnts_pt:
            u_pde_pred_vec, eps_pde_pred_vec = solution(guide_info.x)
            # pde constraint
            sig_pde_pred_vec = mat_model.get_stress(eps_pde_pred_vec)
            sig_pde_pred_tens = mat_model.get_stress_tensor_from_stress_vector(sig_pde_pred_vec)
            div_sig = get_divergence_tensor(sig_pde_pred_tens, guide_info.x, device)
            pde_res += (div_sig - guide_info.source).pow(2).sum(dim=1).mean()
            # compatibility constraint
            eps_from_u_pde = get_gradient_vector(u_pde_pred_vec, guide_info.x)
            eps_pde_pred_tens = mat_model.get_strain_tensor_from_strain_vector(eps_pde_pred_vec)
            compat_res += calc_f_norm((eps_from_u_pde - eps_pde_pred_tens))
            # displacement constraint  
            u_guide_res += ((u_pde_pred_vec - guide_info.disp)/std_u).pow(2).sum(dim=1).mean()
            # strain constraint
            eps_pde_pred_tens = mat_model.get_strain_tensor_from_strain_vector(eps_pde_pred_vec)
            eps_target_tens = mat_model.get_strain_tensor_from_strain_vector(guide_info.strain)
            eps_guide_res += calc_f_norm((eps_pde_pred_tens - eps_target_tens)/std_eps)
        del guide_info
        dbc_res = torch.tensor(0.)
        # dirichlet bc
        for dbc_info in dbc_pnts_pt:
            u_dbc_pred = solution(dbc_info.x)[0]
            dbc_res += ((u_dbc_pred - dbc_info.val)/std_u).pow(2).sum(dim=1).mean()
        del dbc_info
        #
        nbc_res = torch.tensor(0.)
        for nbc_info in nbc_pnts_pt:
            eps_nbc_vec  = solution(nbc_info.x)[1]
            sig_nbc_vec = mat_model.get_stress(eps_nbc_vec)
            sig_nbc_tens = mat_model.get_stress_tensor_from_stress_vector(sig_nbc_vec)
            normal_trac_nbc = torch.einsum("nij,nj->ni", sig_nbc_tens, nbc_info.normal)
            # nuemann bc
            nbc_res += ((normal_trac_nbc - nbc_info.val)).pow(2).sum(dim=1).mean()
        del nbc_info
        # total loss
        objectives = [pde_res , dbc_res , nbc_res , compat_res]
        weights = [weight_pde, weight_dbc, weight_nbc, weight_compat]
        if weight_guide_u > 1e-6:
            objectives.append(u_guide_res)
            weights.append(weight_guide_u)
        if weight_guide_eps > 1e-6:
            objectives.append(eps_guide_res)
            weights.append(weight_guide_eps)
        if need_surgury:
            optimizer.backward_surgery(objectives)
        else:
            optimizer.backward_regular(objectives, weights)
        optimizer.step()
        with torch.no_grad():
            loss_tot = sum([obj.item()*w for obj, w in zip(objectives, weights)])
        # loss_tot = weight_pde * pde_res + weight_dbc * dbc_res + weight_nbc * nbc_res + weight_compat * compat_res +\
        #     weight_guide_u * u_guide_res + weight_guide_eps * eps_guide_res
        # loss_tot.backward()
        # optimizer.step()
        if epoch > lr_sch_epoch:
            optimizer.step_lr_scheduler(loss_tot)
        pde_res_val = -1.
        compat_res_val = -1.
        print(f'epoch {epoch}:', f'loss_tot {loss_tot:.8f} ', 
              f'pde_res {pde_res.item():.8f} dbc_res {dbc_res.item():.8f} ',
              f'nbc_res {nbc_res.item():.8f} compt_res {compat_res.item():.8f} ', 
              f'u_guide_res {u_guide_res.item():.8f} eps_guide_res {eps_guide_res.item():.8f} ',
              f'pde_res_val {pde_res_val:0.8f} compt_res_val {compat_res_val:0.8f}',
        )
        loss_rec['pde'].append(pde_res.item())
        loss_rec['dbc'].append(dbc_res.item())
        loss_rec['nbc'].append(nbc_res.item())
        loss_rec['compat'].append(compat_res.item())
        loss_rec['u_guide'].append(u_guide_res.item())
        loss_rec['eps_guide'].append(eps_guide_res.item())
        loss_rec['pde_val'].append(pde_res_val)
        loss_rec['compat_val'].append(compat_res_val)
        loss_rec['lr'].append(optimizer._optim.param_groups[0]['lr'])
    return loss_rec