from typing import Union
import numpy as np
import torch

class ElasticityLinear():
    def __init__(self, plane_cond, elast_mod, pois_ratio) -> None:
        assert plane_cond in {"plane_stress", "plane_strain"}
        mu = elast_mod/2/(1+pois_ratio)
        lmbda = elast_mod*pois_ratio/(1+pois_ratio)/(1-2*pois_ratio)
        if plane_cond == "plane_stress":
            lmbda = 2*mu*lmbda/(lmbda+2*mu)
        self.Elast_mod = elast_mod
        self.pois_ratio = pois_ratio
        self.mu = mu
        self.lmbda = lmbda

    def get_stress(
        self,
        strain_vector:Union[np.ndarray, torch.Tensor],
        )->Union[np.ndarray, torch.Tensor]:
        """
            strain: (n, 3) e00 e11 e01
        """
        eps_vol = strain_vector[:, 0:1] + strain_vector[:, 1:2]
        sxx = self.lmbda * eps_vol + 2. * self.mu * strain_vector[:, 0:1]
        syy = self.lmbda * eps_vol + 2. * self.mu * strain_vector[:, 1:2]
        sxy = 2. * self.mu * strain_vector[:, 2:3]
        if isinstance(strain_vector, torch.Tensor):
            return torch.concat([sxx, syy, sxy], axis=1)
        elif isinstance(strain_vector, np.ndarray):
            return np.concatenate([sxx, syy, sxy], axis=1)
        else:
            raise NotImplementedError(type(strain_vector))

    def get_stress_tensor_from_stress_vector(
        self,
        stress_vector:Union[np.ndarray, torch.Tensor],
        )->Union[np.ndarray, torch.Tensor]:
        """
            stress: (n, 3) s00 s11 s01
        """
        ndata, ncomp = stress_vector.shape
        assert ncomp == 3, ncomp
        if isinstance(stress_vector, torch.Tensor):
            sig_tensor = torch.zeros(ndata, 2, 2).float().to(device=stress_vector.device)
        elif isinstance(stress_vector, np.ndarray):
            sig_tensor = np.zeros((ndata, 2, 2))
        else:
            raise NotImplementedError(type(stress_vector))
        sig_tensor[:, 0, 0] += stress_vector[:, 0]
        sig_tensor[:, 1, 1] += stress_vector[:, 1]
        sig_tensor[:, 0, 1] += stress_vector[:, 2]
        sig_tensor[:, 1, 0] += stress_vector[:, 2]
        return sig_tensor

    def get_strain_tensor_from_strain_vector(
        self,
        strain_vector:Union[np.ndarray, torch.Tensor],
        )->Union[np.ndarray, torch.Tensor]:
        """
            strain: (n, 3) e00 e11 e01
        """
        return self.get_stress_tensor_from_stress_vector(strain_vector)