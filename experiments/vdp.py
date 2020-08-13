""" Nonlinear example """ 

from systems import VanDerPol
from utils import set_seed
from lib.dlkf import DLKF
from lib.dkf import DKF

from typing import Callable
import numpy as np
import pandas as pd
import pdb
import scipy.stats as stats

import matplotlib.pyplot as plt

set_seed(9001)

dt = 1e-3
T = 10.

mu = lambda t: 3.0
# mu = lambda t: np.cos(t/5) * 3.0
z = VanDerPol(dt, 0.2, mu=mu)
F_hat = z.F
x0 = z.proj(z.x0)
eta0 = np.zeros((z.ndim, z.ndim))
eta = lambda t: eta0

print(F_hat(0))
f1 = DKF(x0, F_hat, z.H, z.Q, z.R, dt)
f2 = DLKF(x0, F_hat, z.H, z.Q, z.R, dt, tau=0.2, gamma=0.5)

max_err = 500.
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

	z_t, x1_t, x2_t = z_t[:z.obs.d], x1_t[:z.obs.d], x2_t[:z.obs.d]

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

fig, axs = plt.subplots(2, 2, figsize=(20, 20))

axs[0,0].plot(hist_z[:,0], hist_z[:,1], color='blue', label='obs')
axs[0,0].plot(hist_f1_x[:,0], hist_f1_x[:,1], color='orange', label='est')
axs[0,0].legend()
axs[0,0].set_title('KF')

axs[1,0].plot(hist_t, hist_f1_res, color='blue')
axs[1,0].set_title('Residual')
axs[1,0].set_ylim((0, 0.2))

axs[0,1].plot(hist_z[:,0], hist_z[:,1], color='blue', label='obs')
axs[0,1].plot(hist_f2_x[:,0], hist_f2_x[:,1], color='orange', label='est')
axs[0,1].legend()
axs[0,1].set_title('LKF')

axs[1,1].plot(hist_t, hist_f2_res, color='blue')
axs[1,1].set_title('Residual')
axs[1,1].set_ylim((0, 0.2))

plt.tight_layout()
plt.show()
