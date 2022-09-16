from typing import Tuple
import os
import pathlib

pjoin = os.path.join
SCRIPT_DIR = os.path.abspath(pathlib.Path(__file__).parent.absolute())
PROBLEM_SETUP_DIR = pjoin(SCRIPT_DIR, "data")
PLANE_COND = "plane_strain"
ELAST_MOD = 1.
POISSON_RATIO = 0.3
HOLE_RAD = 0.25
SIG_FAR = 0.2
SQUARE_LENGTH = 0.5
BOUNDARY_SEG_SZ_TRAIN = 0.25 / 101
BOUNDARY_SEG_SZ_VALID = 0.25 / 27
BOUNDARY_SAMPLE_TYPE = 'uniform'
NUM_TETA_PDE_PNTS_TRAIN = 50
NUM_RAD_PDE_PNTS_TRAIN = 50
NUM_TETA_PDE_PNTS_VALID = 23
NUM_RAD_PDE_PNTS_VALID = 23
PDE_SAMPLE_TYPE = 'uniform'


import pandas as pd
import numpy as np
from pyDOE import lhs
PI = np.pi

class ExactSolution(object):
    def __init__(self):
        import sympy as sp 
        from sympy.abc import x, y
        E, nu = ELAST_MOD, POISSON_RATIO
        a, sig0 = HOLE_RAD, SIG_FAR

        r = sp.sqrt(x**2 + y**2)
        teta = sp.atan2(y, x)
        fac = sig0 * (1. + nu) * a / E / 2.
        sin, cos = sp.sin, sp.cos
        ux = 2.*(1.-nu)*(r/a+2.*a/r)*cos(teta) + (a/r-a**3/r**3)*cos(3.*teta)
        ux *= fac
        uy = -2.*(1.-2.*nu)*a/r*sin(teta) - 2.*nu*r/a*sin(teta) + (a/r - a**3/r**3)*sin(3*teta)
        uy *= fac
        sxx = 1. + (1.5*a**4/r**4 - a**2/r**2) * cos(4.*teta) - 1.5 * a**2 / r**2 * cos(2.* teta)
        sxx *= sig0
        syy = -(1.5*a**4/r**4 - a**2/r**2) * cos(4.*teta) - 0.5 * a**2 / r**2 * cos(2.* teta)
        syy *= sig0
        sxy = (1.5*a**4/r**4 - a**2/r**2) * sin(4.*teta) - 0.5 * a**2 / r**2 * sin(2.* teta)
        sxy *= sig0
        
        exx = sp.diff(ux, x)
        eyy = sp.diff(uy, y)
        exy = (sp.diff(ux, y) + sp.diff(uy, x)) * 0.5

        self.ux = sp.lambdify([x, y], ux, "numpy")
        self.uy = sp.lambdify([x, y], uy, "numpy")
        self.exx = sp.lambdify([x, y], exx, "numpy")
        self.eyy = sp.lambdify([x, y], eyy, "numpy")
        self.exy = sp.lambdify([x, y], exy, "numpy")
        self.sxx = sp.lambdify([x, y], sxx, "numpy")
        self.syy = sp.lambdify([x, y], syy, "numpy")
        self.sxy = sp.lambdify([x, y], sxy, "numpy")
    
    def get_u(self, x: np.ndarray) -> np.ndarray:
        assert x.ndim == 2
        assert x.shape[1] == 2
        u0 = self.ux(x[:, 0:1], x[:, 1:2])
        if isinstance(u0, int) or isinstance(u0, float):
            print(20*"!")
            u0 = u0 * np.ones((x.shape[0], 1))
        u1 = self.uy(x[:, 0:1], x[:, 1:2])
        if isinstance(u1, int) or isinstance(u1, float):
            print(20*"!")
            u1 = u1 * np.ones((x.shape[0], 1))
        u = np.hstack((u0, u1))
        return u

    def get_eps(self, cord:np.ndarray)->np.ndarray:
        assert cord.ndim == 2
        assert cord.shape[1] == 2
        x, y = cord[:, 0:1], cord[:, 1:2]
        exx = self.exx(x, y)
        if isinstance(exx, int) or isinstance(exx, float):
            print(20*"!")
            exx = exx * np.ones_like(x)
        eyy = self.eyy(x, y)
        if isinstance(eyy, int) or isinstance(eyy, float):
            print(20*"!")
            eyy = eyy * np.ones_like(y)
        exy = self.exy(x, y)
        if isinstance(exy, int) or isinstance(exy, float):
            print(20*"!")
            exy = exy * np.ones_like(y)
        return np.hstack((exx, eyy, exy))

    def get_sig(self, x, y):
        assert x.ndim == 2
        assert y.ndim == 2
        assert x.shape[1] == 1
        assert y.shape[1] == 1
        sxx = self.sxx(x, y)
        if isinstance(sxx, int) or isinstance(sxx, float):
            print(20*"!")
            sxx = sxx * np.ones_like(x)
        syy = self.syy(x, y)
        if isinstance(syy, int) or isinstance(syy, float):
            print(20*"!")
            syy = syy * np.ones_like(y)
        sxy = self.exy(x, y)
        if isinstance(sxy, int) or isinstance(sxy, float):
            print(20*"!")
            sxy = sxy * np.ones_like(y)
        sig = np.hstack((sxx, syy, sxy))
        return sig

def sample_dirichlet_boundary(seg_size)->np.ndarray:
    all_cords = []
    # bottom
    N = int((SQUARE_LENGTH-HOLE_RAD) // seg_size + 1)
    if BOUNDARY_SAMPLE_TYPE == 'random':
        temp = np.random.uniform(HOLE_RAD, SQUARE_LENGTH, N)
    elif BOUNDARY_SAMPLE_TYPE == 'uniform':
        temp = np.linspace(HOLE_RAD, SQUARE_LENGTH, N)
    else:
        raise NotImplementedError()
    cord = np.zeros((N, 2))
    cord[:, 0] = temp
    all_cords.append(cord)
    # right
    N = int(SQUARE_LENGTH // seg_size + 1)
    if BOUNDARY_SAMPLE_TYPE == 'random':
        temp = np.random.uniform(0., SQUARE_LENGTH, N)
    elif BOUNDARY_SAMPLE_TYPE == 'uniform':
        temp = np.linspace(0., SQUARE_LENGTH, N)[1:]
    else:
        raise NotImplementedError()
    cord = np.zeros((len(temp), 2))
    cord[:, 0] = SQUARE_LENGTH
    cord[:, 1] = temp
    all_cords.append(cord)
    # top
    N = int(SQUARE_LENGTH // seg_size + 1)
    if BOUNDARY_SAMPLE_TYPE == 'random':
        temp = np.random.uniform(0., SQUARE_LENGTH, N)
    elif BOUNDARY_SAMPLE_TYPE == 'uniform':
        temp = np.linspace(SQUARE_LENGTH, 0., N)[1:]
    else:
        raise NotImplementedError()
    cord = np.zeros((len(temp), 2))
    cord[:, 0] = temp
    cord[:, 1] = SQUARE_LENGTH
    all_cords.append(cord)
    # left
    N = int((SQUARE_LENGTH-HOLE_RAD) // seg_size + 1)
    if BOUNDARY_SAMPLE_TYPE == 'random':
        temp = np.random.uniform(HOLE_RAD, SQUARE_LENGTH, N)
    elif BOUNDARY_SAMPLE_TYPE == 'uniform':
        temp = np.linspace(SQUARE_LENGTH, HOLE_RAD, N)[1:]
    else:
        raise NotImplementedError()
    cord = np.zeros((len(temp), 2))
    cord[:, 1] = temp
    all_cords.append(cord)
    all_cords = np.concatenate(all_cords, axis=0)
    return all_cords



def sample_neumann_boundary(seg_size)->Tuple[np.ndarray, np.ndarray]:
    # quarter of a circle
    N = int(np.pi*HOLE_RAD/2./seg_size +1)
    if BOUNDARY_SAMPLE_TYPE == 'random':
        temp = np.random.uniform(0., np.pi/2, N)
    elif BOUNDARY_SAMPLE_TYPE == 'uniform':
        temp = np.linspace(0., np.pi/2, N)[1:-1]
    else:
        raise NotImplementedError()
    cord = np.zeros((len(temp), 2))
    cord[:, 0] = HOLE_RAD * np.cos(temp)
    cord[:, 1] = HOLE_RAD * np.sin(temp)
    normal = np.zeros((len(temp), 2))
    normal[:, 0] = -np.cos(temp)
    normal[:, 1] = -np.sin(temp)
    return cord, normal


def sample_pde_points(num_teta, num_rad, d_type='uniform'):
    if d_type == 'uniform':
        tets = np.linspace(0., PI/4., num_teta)
        sin, cos = np.sin, np.cos
        col_pnts = []
        for i, tet in enumerate(tets):
            pnts = np.linspace(0., SQUARE_LENGTH, num_rad)[1:-1].reshape(-1, 1)
            pnts = np.hstack((pnts, np.zeros_like(pnts)))

            rot = np.array([[cos(tet), -sin(tet)], [sin(tet), cos(tet)]])
            l = SQUARE_LENGTH / cos(tet) - HOLE_RAD
            pnts = pnts @ rot.T * l
            pnts[:, 0] += HOLE_RAD * cos(tet)
            pnts[:, 1] += HOLE_RAD * sin(tet)
            col_pnts.append(pnts)
        tets = np.linspace(PI/4., PI/2, num_teta)[1:]
        for i, tet in enumerate(tets):
            pnts = np.linspace(0., SQUARE_LENGTH, num_rad)[1:-1].reshape(-1, 1)
            pnts = np.hstack((pnts, np.zeros_like(pnts)))

            rot = np.array([[cos(tet), -sin(tet)], [sin(tet), cos(tet)]])
            l = SQUARE_LENGTH / sin(tet) - HOLE_RAD
            pnts = pnts @ rot.T * l
            pnts[:, 0] += HOLE_RAD * cos(tet)
            pnts[:, 1] += HOLE_RAD * sin(tet)
            col_pnts.append(pnts)
        col_pnts = np.concatenate(col_pnts)
    elif d_type == 'random':
        N = num_teta * num_rad
        np.random.seed(0)
        col_pnts = lhs(2, N) * SQUARE_LENGTH
        np.random.seed(None)
        target_ids = (col_pnts[:, 0]**2 + col_pnts[:, 1]**2 > HOLE_RAD**2)
        col_pnts = col_pnts[target_ids, :]
    else:
        raise NotImplementedError()
    assert col_pnts.ndim == 2
    assert col_pnts.shape[1] == 2
    return col_pnts

if __name__ == "__main__":
    if not os.path.exists(PROBLEM_SETUP_DIR):
        os.makedirs(PROBLEM_SETUP_DIR)
    sol = ExactSolution()
    # train data
    cord_pde_train = sample_pde_points(
        NUM_TETA_PDE_PNTS_TRAIN, NUM_RAD_PDE_PNTS_TRAIN, PDE_SAMPLE_TYPE
    )
    u_pde_train = sol.get_u(cord_pde_train)
    eps_pde_train = sol.get_eps(cord_pde_train)
    cord_dbc_train = sample_dirichlet_boundary(BOUNDARY_SEG_SZ_TRAIN)
    u_dbc_train = sol.get_u(cord_dbc_train)
    cord_nbc_train, normal_nbc_train = sample_neumann_boundary(BOUNDARY_SEG_SZ_TRAIN)
    trac_nbc_train = np.zeros_like(cord_nbc_train)
    # validation data
    cord_pde_valid = sample_pde_points(
        NUM_TETA_PDE_PNTS_VALID, NUM_RAD_PDE_PNTS_VALID, PDE_SAMPLE_TYPE
    )
    u_pde_valid = sol.get_u(cord_pde_valid)
    eps_pde_valid = sol.get_eps(cord_pde_valid)
    cord_dbc_valid = sample_dirichlet_boundary(BOUNDARY_SEG_SZ_VALID)
    u_dbc_valid = sol.get_u(cord_dbc_valid)
    cord_nbc_valid, normal_nbc_valid = sample_neumann_boundary(BOUNDARY_SEG_SZ_VALID)
    trac_nbc_valid = np.zeros_like(cord_nbc_valid)
    # save the data
    pd.DataFrame(
        {'x0':cord_pde_train[:, 0], 'x1':cord_pde_train[:, 1],
         'u0':u_pde_train[:, 0], 'u1':u_pde_train[:, 1],
         'e00':eps_pde_train[:, 0], 'e11':eps_pde_train[:, 1],
         'e01':eps_pde_train[:, 2],
         'source0':np.zeros(eps_pde_train.shape[0]),
         'source1':np.zeros(eps_pde_train.shape[0])
        }
    ).to_csv(pjoin(PROBLEM_SETUP_DIR, "pde_data_train.csv"), index=False)
    pd.DataFrame(
        {'x0':cord_dbc_train[:, 0], 'x1':cord_dbc_train[:, 1],
         'u0':u_dbc_train[:, 0], 'u1':u_dbc_train[:, 1]
        }
    ).to_csv(pjoin(PROBLEM_SETUP_DIR, "dirichlet_bc_data_train.csv"), index=False)
    pd.DataFrame(
        {'x0':cord_nbc_train[:, 0], 'x1':cord_nbc_train[:, 1],
         'n0':normal_nbc_train[:, 0], 'n1':normal_nbc_train[:, 1],
         'trac0':trac_nbc_train[:, 0], 'trac1':trac_nbc_train[:, 1]
        }
    ).to_csv(pjoin(PROBLEM_SETUP_DIR, "neumann_bc_data_train.csv"), index=False)    



    pd.DataFrame(
        {'x0':cord_pde_valid[:, 0], 'x1':cord_pde_valid[:, 1],
         'u0':u_pde_valid[:, 0], 'u1':u_pde_valid[:, 1],
         'e00':eps_pde_valid[:, 0], 'e11':eps_pde_valid[:, 1],
         'e01':eps_pde_valid[:, 2]
        }
    ).to_csv(pjoin(PROBLEM_SETUP_DIR, "pde_data_validation.csv"), index=False)

    pd.DataFrame(
        {'x0':cord_dbc_valid[:, 0], 'x1':cord_dbc_valid[:, 1],
         'u0':u_dbc_valid[:, 0], 'u1':u_dbc_valid[:, 1]
        }
    ).to_csv(pjoin(PROBLEM_SETUP_DIR, "dirichlet_bc_data_validation.csv"), index=False)

    pd.DataFrame(
        {'x0':cord_nbc_valid[:, 0], 'x1':cord_nbc_valid[:, 1],
         'n0':normal_nbc_valid[:, 0], 'n1':normal_nbc_valid[:, 1],
         'trac0':trac_nbc_valid[:, 0], 'trac1':trac_nbc_valid[:, 1]
        }
    ).to_csv(pjoin(PROBLEM_SETUP_DIR, "neumann_bc_data_validation.csv"), index=False)