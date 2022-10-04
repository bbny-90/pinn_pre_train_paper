import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import rc
rc('font',size=18)
rc('font',family='serif')
rc('axes',labelsize=20)
rc('lines', linewidth=2,markersize=10)

def contour(
    x:np.ndarray, 
    y:np.ndarray, 
    z:np.ndarray, 
    title:str, 
    dir_save:str, 
    name_save:str, 
    vmax=None,
    vmin=None,
    levels=50):
    """
    Contour plot.
        x: Nx by Ny arraye
        y: Nx by Ny arraye
        z: Nx by Ny arraye
        levels: number of contour lines.
    """
    if vmin is None: vmin = np.min(z)
    if vmax is None: vmax = np.max(z)
    plt.contour(x, y, z, colors='k', linewidths=0.2, levels=levels)
    plt.contourf(x, y, z, cmap='rainbow', levels=levels, norm=Normalize(vmin=vmin, vmax=vmax))
    plt.title(title)
    cbar = plt.colorbar(pad=0.03, aspect=25, format='%.0e')
    cbar.mappable.set_clim(vmin, vmax)
    plt.tight_layout()
    plt.savefig(dir_save + name_save + ".png")
    plt.close()