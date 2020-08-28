''' Learning Koopman-operator based Kalman filter for identifying nonlinear systems
'''

from typing import Callable
import numpy as np
import pandas as pd
import pdb
import scipy.stats as stats

from utils import *
from totorch.operators import Koopman

class LKKF:
	def __init__(self, 
		x0: np.ndarray, K: Koopman, H: np.ndarray, Q: np.ndarray, R: np.ndarray, 	# KF parameters
		dt: float, tau=float('inf'), eps=1e-4, gamma=1.								# Hyperparameters
	):
		self.F = lambda x: (K@x - x) / dt # Approximate differential model with forward-difference
		self.H = H
		self.Q = Q
		self.R = R
		self.dt = dt
		self.tau = tau
		self.eps = eps
		self.gamma = gamma
		self.ndim = x0.shape[0]

		# self.P_act_till = np.zeros((self.tau_n, self.ndim, self.ndim))

		# Initial conditions
		self.t = 0.
		self.x_t = x0.copy()[:, np.newaxis]
		self.P_t = np.eye(self.ndim)
		self.eta_t = np.zeros((self.ndim, self.ndim))

		# Memory
		self.err_hist = []
		self.P_hist = []

	def step(self, z_t):
		x_t, P_t, H, Q, R, tau = self.x_t, self.P_t, self.H, self.Q, self.R, self.tau
		z_t = z_t[:, np.newaxis]
		K_t = P_t@H@np.linalg.inv(R) 

		if self.t > self.tau: # TODO: warm start?
			# Method 1
			err_t, err_tau = self.err_hist[-1][:,np.newaxis], self.err_hist[0][:,np.newaxis]
			P_tau = self.P_hist[0]
			act_dPdt = (err_t@err_t.T - err_tau@err_tau.T) / tau 
			est_dPdt = H@(P_t - P_tau)@H.T / tau

			# Method 2
			# for i in range(self.tau_n):
			# 	self.P_act_till[i] = np.outer(self.err_hist[-self.tau_n+i], self.err_hist[-self.tau_n+i])
			# act_Pdt = np.gradient(self.P_act_till, self.dt, axis=0).mean(axis=0)
			# est_Pdt = np.gradient(self.P_hist[-self.tau_n:], self.dt, axis=0).mean(axis=0)

			C_t = act_dPdt - est_dPdt 

			# Method 1
			H_inv = np.linalg.inv(H)
			P_inv = pinv(P_t, eps=self.eps)
			self.eta_t = self.gamma * H_inv@C_t@H_inv.T@P_inv / 2

			# Method 2
			# C_inv_t = np.linalg.inv(C_t)
			# self.eta_t = self.gamma / 2 * pinv(P_t@H.T@C_inv_t@H, eps=self.eps)

		F_est = lambda x: self.F(x) - self.eta_t@x # TODO: re-project eta?
		dx_dt = F_est(x_t) + K_t@(z_t - H@x_t) # TODO: re-project?
		dP_dt = F_est(P_t) + F_est(P_t.T).T + Q - K_t@R@K_t.T # TODO: re-project?
		self.t += self.dt
		self.x_t += dx_dt * self.dt
		self.P_t += dP_dt * self.dt

	def __call__(self, z_t: np.ndarray):
		''' Observe through filter ''' 
		self.step(z_t)
		err_t = z_t - np.squeeze(self.x_t)@self.H.T
		self.P_hist.append(self.P_t)
		self.err_hist.append(err_t)
		if self.t > self.tau:
			del self.P_hist[0]
			del self.err_hist[0]
		return self.x_t.copy(), err_t 

if __name__ == '__main__':
	import matplotlib.pyplot as plt

	set_seed(4001)

	dt = 1e-5
	n = 2000000

	""" Noisy LTI example """ 
	# z = Oscillator(dt, 0.0, 1.0)
	# z = SpiralSink(dt, 0.0, 1.0)
	# eta_mu, eta_var = 0., 0.1
	# eta0 = np.random.normal(eta_mu, eta_var, (2, 2))
	# eta = lambda t: eta0
	# F_hat = lambda t: z.F(t) + eta(t)

	""" Partially known LTV example """ 
	z = TimeVarying(dt, 0.0, 1.0, f=1/20)
	F_hat = lambda t: z.F(0)
	eta = lambda t: F_hat(t) - z.F(t)
	# eta_bnd = 10*max(np.linalg.norm(z.F1 - z.F0), np.linalg.norm(z.F2 - z.F0))
	eta_bnd = float('inf')

	print(F_hat(0))
	f = LKF(z.x0, F_hat, z.H, z.Q, z.R, dt, tau=0.1, eta_bnd=eta_bnd, eps= 3e-3)

	max_err = 10.
	max_eta_err = 100
	max_zz = 100. 

	hist_t = []
	hist_z = []
	hist_x = []
	hist_err = []
	hist_eta = []
	hist_ezz = []
	hist_pin = []
	hist_p = []
	for _ in range(n):
		z_t = z()
		x_t, err_t = f(z_t)
		hist_z.append(z_t)
		hist_t.append(z.t)
		hist_x.append(x_t) 
		hist_err.append(err_t)
		hist_eta.append(f.eta_t.copy()) # variation
		hist_ezz.append(f.C_t.copy())
		hist_pin.append(f.P_inv_t.copy())
		hist_p.append(f.P_t.copy())

		# Error condition 1
		if np.linalg.norm(err_t) > max_err:
			print('Error overflowed!')
			break

		# Error condition 2
		if np.linalg.norm(f.eta_t - eta(z.t)) > max_eta_err:
			print('Variation error overflowed!')
			break

		# Error condition 3
		if np.linalg.norm(f.C_t) > max_zz:
			print('d_zz overflowed!')
			break

	# start, end = None, 20000 # for case analysis
	start, end = None, None # for case analysis
	every = 100

	hist_t = np.array(hist_t)[start:end:every]
	hist_z = np.array(hist_z)[start:end:every]
	hist_x = np.array(hist_x)[start:end:every]
	hist_err = np.array(hist_err)[start:end:every]
	hist_eta = np.array(hist_eta)[start:end:every]
	hist_ezz = np.array(hist_ezz)[start:end:every]
	hist_pin = np.array(hist_pin)[start:end:every]
	hist_p = np.array(hist_p)[start:end:every]

	# pdb.set_trace()

	fig, axs = plt.subplots(3, 4, figsize=(20, 20))
	fig.suptitle('LKF')
	
	axs[0,0].plot(hist_z[:,0], hist_z[:,1], color='blue', label='obs')
	axs[0,0].plot(hist_x[:,0], hist_x[:,1], color='orange', label='est')
	axs[0,0].legend()
	axs[0,0].set_title('System')

	axs[0,1].plot(hist_t, hist_z[:,0], color='blue', label='obs')
	axs[0,1].plot(hist_t, hist_x[:,0], color='orange', label='est')
	axs[0,1].set_title('Axis 1')

	axs[0,2].plot(hist_t, hist_z[:,1], color='blue', label='obs')
	axs[0,2].plot(hist_t, hist_x[:,1], color='orange', label='est')
	axs[0,2].set_title('Axis 2')

	var_err = hist_eta - np.array(list(map(eta, hist_t)))
	axs[1,0].plot(hist_t, np.linalg.norm(var_err, axis=(1,2)))
	axs[1,0].set_title('Variation error (norm)')

	axs[1,1].plot(hist_t, hist_err[:,0])
	axs[1,1].set_title('Axis 1 error')

	axs[1,2].plot(hist_t, hist_err[:,1])
	axs[1,2].set_title('Axis 2 error')

	var_err_rast = var_err.reshape((var_err.shape[0], 4))
	axs[1,3].plot(hist_t, var_err_rast[:,0])
	axs[1,3].plot(hist_t, var_err_rast[:,1])
	axs[1,3].plot(hist_t, var_err_rast[:,2])
	axs[1,3].plot(hist_t, var_err_rast[:,3])
	axs[1,3].set_title('Variation error (rasterized)')

	var_err_bs = -np.array(list(map(eta, hist_t)))
	var_err_rast_bs = var_err_bs.reshape((var_err.shape[0], 4))
	axs[2,0].plot(hist_t, var_err_rast_bs[:,0])
	axs[2,0].plot(hist_t, var_err_rast_bs[:,1])
	axs[2,0].plot(hist_t, var_err_rast_bs[:,2])
	axs[2,0].plot(hist_t, var_err_rast_bs[:,3])
	axs[2,0].set_title('Variation error (baseline, rasterized)')

	axs[2,1].plot(hist_t, np.linalg.norm(hist_ezz, axis=(1,2)))
	axs[2,1].set_title('d_zz norm')

	p_inv_rast = hist_pin.reshape((hist_pin.shape[0], 4))
	axs[2,2].plot(hist_t, p_inv_rast[:,0])
	axs[2,2].plot(hist_t, p_inv_rast[:,1])
	axs[2,2].plot(hist_t, p_inv_rast[:,2])
	axs[2,2].plot(hist_t, p_inv_rast[:,3])
	axs[2,2].set_title('P_inv (rasterized)')

	p_rast = hist_p.reshape((hist_p.shape[0], 4))
	axs[2,3].plot(hist_t, p_rast[:,0])
	axs[2,3].plot(hist_t, p_rast[:,1])
	axs[2,3].plot(hist_t, p_rast[:,2])
	axs[2,3].plot(hist_t, p_rast[:,3])
	axs[2,3].set_title('P (rasterized)')


	# axs[3].plot(hist_t, hist_eta)

	# compact plot
	# fig, axs = plt.subplots(1, 2, figsize=(10, 5))
	# axs[0].plot(hist_z[:,0], hist_z[:,1], color='blue', label='obs')
	# axs[0].plot(hist_x[:,0], hist_x[:,1], color='orange', label='est')
	# axs[0].legend()
	# axs[0].set_title('System')
	# axs[1].plot(hist_t, hist_err[:,0])
	# axs[1].set_title('Axis 1 error')

	plt.show()
