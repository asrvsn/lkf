""" Nonlinear example """ 

from systems import Lorenz
from utils import set_seed
from lib.lkf import LKF
from lib.kf import KF

from typing import Callable
import numpy as np
import pandas as pd
import pdb
import scipy.stats as stats

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 

set_seed(9001)

dt = 5e-3
T = 20.

z = Lorenz(dt, 0.1)
F_hat = z.F
x0 = z.proj(z.x0)
eta0 = np.zeros((z.ndim, z.ndim))
eta = lambda t: eta0

print(F_hat(0))
f1 = KF(x0, F_hat, z.H, z.Q, z.R, dt)
f2 = LKF(x0, F_hat, z.H, z.Q, z.R, dt, tau=0.15, eps=1e-2, gamma=0.25)

max_err = 50.
max_eta_err = 100
max_zz = 1000. 

hist_t = []
hist_z = []
hist_f1_x = []
hist_f2_x = []
hist_f1_err = []
hist_f2_err = []
hist_eta = []
while z.t <= T:
	z_t = z()
	x1_t, err1_t = f1(z_t)
	x2_t, err2_t = f2(z_t)

	z_t, x1_t, err1_t, x2_t, err2_t = z_t[:z.obs.d], x1_t[:z.obs.d], err1_t[:z.obs.d], x2_t[:z.obs.d], err2_t[:z.obs.d] 

	hist_z.append(z_t)
	hist_t.append(z.t)
	hist_f1_x.append(x1_t) 
	hist_f2_x.append(x2_t) 
	hist_f1_err.append(err1_t)
	hist_f2_err.append(err2_t)
	hist_eta.append(f2.eta_t.copy())

	# Error condition 1
	if np.linalg.norm(err2_t) > max_err:
		print('Error overflowed!')
		break

	# # Error condition 2
	# if np.linalg.norm(f2.eta_t - eta(z.t)) > max_eta_err:
	# 	print('Variation error overflowed!')
	# 	break

	# # Error condition 3
	# if np.linalg.norm(f2.C_t) > max_zz:
	# 	print('d_zz overflowed!')
	# 	break

# start, end = None, 20000 # for case analysis
start, end = None, None # for case analysis
every = 1

hist_t = np.array(hist_t)[start:end:every]
hist_z = np.array(hist_z)[start:end:every]
hist_f1_x = np.array(hist_f1_x)[start:end:every]
hist_f2_x = np.array(hist_f2_x)[start:end:every]
hist_f1_err = np.array(hist_f1_err)[start:end:every]
hist_f1_res = np.linalg.norm(hist_f1_err, axis=1)
hist_f2_err = np.array(hist_f2_err)[start:end:every]
hist_f2_res = np.linalg.norm(hist_f2_err, axis=1)
hist_eta = np.array(hist_eta)[start:end:every]

# pdb.set_trace()

fig = plt.figure()

ax = fig.add_subplot(2, 2, 1, projection='3d')
ax.plot(hist_z[:,0], hist_z[:,1], hist_z[:,2], color='blue', label='obs')
ax.plot(hist_f1_x[:,0], hist_f1_x[:,1], hist_f1_x[:,2], color='orange', label='est')
ax.legend()
ax.set_title('KF')

ax = fig.add_subplot(2, 2, 3)
ax.plot(hist_t, hist_f1_res, color='blue')
ax.set_title('Residual')

ax = fig.add_subplot(2, 2, 2, projection='3d')
ax.plot(hist_z[:,0], hist_z[:,1], hist_z[:,2], color='blue', label='obs')
ax.plot(hist_f2_x[:,0], hist_f2_x[:,1], hist_f2_x[:,2], color='orange', label='est')
ax.legend()
ax.set_title('LKF')

ax = fig.add_subplot(2, 2, 4)
ax.plot(hist_t, hist_f2_res, color='blue')
ax.set_title('Residual')

plt.tight_layout()
plt.show()
