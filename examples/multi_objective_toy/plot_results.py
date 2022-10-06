import os
import pathlib
import numpy as np
import pandas as pd
import torch
pjoin = os.path.join
SCRIPT_DIR = os.path.abspath(pathlib.Path(__file__).parent.absolute())

from pdes.operators import get_grad_scalar
from examples.multi_objective_toy.main import get_obj1, get_obj2

import matplotlib.pyplot as plt
import matplotlib.animation as animation 

x0 = np.linspace(-4., 2., 1000)
x1 = np.linspace(-4., 7, 1000)
x0, x1 = np.meshgrid(x0, x1)
x = np.stack([x0.flatten(), x1.flatten()], axis=1)
x = torch.from_numpy(x).float().requires_grad_(True)
obj1 = get_obj1(x).view(-1, 1)
grad_obj1 = get_grad_scalar(obj1, x).detach().numpy()
obj2 = get_obj2(x).view(-1, 1)
grad_obj2 = get_grad_scalar(obj2, x).detach().numpy()

x = x.detach().numpy()
obj1 = obj1.detach().numpy().flatten()
obj2 = obj2.detach().numpy().flatten()
tot_loss = obj1 + obj2

vanilla_df = pd.read_csv(pjoin(SCRIPT_DIR, '.tmp/vanilla.csv'))
pcg_df = pd.read_csv(pjoin(SCRIPT_DIR, '.tmp/grad_surg.csv'))
vmin, vmax = -580, -10
levels = 40
fig,ax = plt.subplots()
contourf_ = ax.contourf(x0, x1, tot_loss.reshape(x0.shape), levels=levels)
cbar = fig.colorbar(contourf_)
plt.scatter(pcg_df['p0'], pcg_df['p1'], c='r', s=1)
plt.savefig(pjoin(SCRIPT_DIR, ".tmp/grad_surg_conv.png"))
plt.show()

fig,ax = plt.subplots()
contourf_ = ax.contourf(x0, x1, tot_loss.reshape(x0.shape), levels=levels)
cbar = fig.colorbar(contourf_)
plt.scatter(vanilla_df['p0'], vanilla_df['p1'], c='r', s=1.5)
plt.savefig(pjoin(SCRIPT_DIR, ".tmp/vanilla_conv.png"))
plt.show()

#Plot the contour
vmin, vmax = -580, -10
levels = 40
fig,ax = plt.subplots()
contourf_ = ax.contourf(x0, x1, tot_loss.reshape(x0.shape), levels=levels)
cbar = fig.colorbar(contourf_)
step = 300


# Create animation
line, = ax.plot([], [], 'r', label = 'PCGrad Adam', lw = 1.5)
point, = ax.plot([], [], '*', color = 'red', markersize = 4)

vector1 = ax.quiver([0.], [0.], [1.], [1.], color='b')
vector2 = ax.quiver([0.], [0.], [1.], [1.], color='purple')
value_display = ax.text(0.02, 0.02, '', transform=ax.transAxes)

def init_1():
    line.set_data([], [])
    point.set_data([], [])
    value_display.set_text('')

    return line, point#, value_display

def animate_1(i):
    # Animate line
    line.set_data(pcg_df['p0'][0:i:step], pcg_df['p1'][0:i:step])
    pnt = pcg_df[['p0', 'p1']].iloc[i].to_numpy()
    # Animate points
    point.set_data(pnt[0], pnt[1])
    vector1.set_offsets(pnt)
    vector2.set_offsets(pnt)
    xx = torch.from_numpy(pnt).float().view(1, -1).requires_grad_()
    grad1 = -get_grad_scalar(get_obj1(xx).view(1, -1), xx).detach().numpy().flatten()
    grad2 = -get_grad_scalar(get_obj2(xx).view(1, -1), xx).detach().numpy().flatten()
    vector1.set_UVC([grad1[0]], [grad1[1]])
    vector1.set_UVC([grad2[0]], [grad2[1]])

    # # Animate value display
    value_display.set_text('Iteration = ' + str(i))

    return line, point, value_display

ax.legend(loc = 1)

anim1 = animation.FuncAnimation(fig, animate_1, init_func=init_1,
                               frames=range(0, len(pcg_df['p0']), step), interval=1, 
                               repeat_delay=60, blit=False)
plt.show()
writergif = animation.PillowWriter(fps=30)
anim1.save(pjoin(SCRIPT_DIR, '.tmp/pcgrad.gif'), writer=writergif)
plt.close()


fig,ax = plt.subplots()
contourf_ = ax.contourf(x0, x1, tot_loss.reshape(x0.shape), levels=levels)
cbar = fig.colorbar(contourf_)
step = 300


# Create animation
line, = ax.plot([], [], 'r', label = 'Vanilla Adam', lw = 1.5)
point, = ax.plot([], [], '*', color = 'red', markersize = 4)
value_display = ax.text(0.02, 0.02, '', transform=ax.transAxes)
vector1 = ax.quiver([0.], [0.], [1.], [1.], color='b')
vector2 = ax.quiver([0.], [0.], [1.], [1.], color='purple')
value_display = ax.text(0.02, 0.02, '', transform=ax.transAxes)

def init_1():
    line.set_data([], [])
    point.set_data([], [])
    value_display.set_text('')

    return line, point#, value_display

def animate_1(i):
    # Animate line
    line.set_data(vanilla_df['p0'][0:i:step], vanilla_df['p1'][0:i:step])
    pnt = vanilla_df[['p0', 'p1']].iloc[i].to_numpy()
    # Animate points
    point.set_data(pnt[0], pnt[1])
    vector1.set_offsets(pnt)
    vector2.set_offsets(pnt)
    xx = torch.from_numpy(pnt).float().view(1, -1).requires_grad_()
    grad1 = -get_grad_scalar(get_obj1(xx).view(1, -1), xx).detach().numpy().flatten()
    grad2 = -get_grad_scalar(get_obj2(xx).view(1, -1), xx).detach().numpy().flatten()
    vector1.set_UVC([grad1[0]], [grad1[1]])
    vector1.set_UVC([grad2[0]], [grad2[1]])
    # # Animate value display
    value_display.set_text('Iteration = ' + str(i))

    return line, point, value_display

ax.legend(loc = 1)

anim1 = animation.FuncAnimation(fig, animate_1, init_func=init_1,
                               frames=range(0, len(vanilla_df['p0']), step), interval=1, 
                               repeat_delay=60, blit=False)
plt.show()
writergif = animation.PillowWriter(fps=30)
anim1.save(pjoin(SCRIPT_DIR, '.tmp/vanilla.gif'), writer=writergif)
plt.close()