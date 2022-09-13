import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import rc
rc('font',size=18)
rc('font',family='serif')
rc('axes',labelsize=20)
rc('lines', linewidth=2,markersize=10)

def contour(x, y, z, title, dir_save, name_save, levels=50):
    """
    Contour plot.
        x: x-array.
        y: y-array.
        z: z-array.
        levels: number of contour lines.
    """
    vmin = np.min(z)
    vmax = np.max(z)
    plt.contour(x, y, z, colors='k', linewidths=0.2, levels=levels)
    plt.contourf(x, y, z, cmap='rainbow', levels=levels, norm=Normalize(vmin=vmin, vmax=vmax))
    plt.title(title)
    cbar = plt.colorbar(pad=0.03, aspect=25, format='%.0e')
    cbar.mappable.set_clim(vmin, vmax)
    plt.tight_layout()
    plt.savefig(dir_save + name_save + ".png")
    plt.show()
    plt.close()