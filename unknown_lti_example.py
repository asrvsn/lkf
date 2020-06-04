''' Learning Kalman-Bucy filter
'''

from systems import *
from integrator import Integrator
from utils import set_seed
from lkf import LKF
from kf import KF

from typing import Callable
import numpy as np
import pandas as pd
import pdb
import scipy.stats as stats

class LKF(LSProcess):
	def __init__(self, x0: np.ndarray, F: Callable, H: np.ndarray, Q: np.ndarray, R: np.ndarray, dt: float, tau: float = float('inf'), eta_bnd: float = float('inf'), eps=1e-4):
		self.F = F
		self.H = H
		self.Q = Q
		self.R = R
		self.dt = dt
		self.tau = tau
		self.eps = eps
		self.eta_bnd = eta_bnd
		self.ndim = x0.shape[0]

		self.err_hist = []
		self.e_zz_t = np.zeros((self.ndim, self.ndim)) # temp var..
		self.p_inv_t = np.zeros((self.ndim, self.ndim)) # temp var..

		def f(t, state, z_t, err_hist, F_t):
			# TODO fix all stateful references in this body

			state = state.reshape(self.ode_shape)
			x_t, P_t, eta_t = state[:, :1], state[:, 1:1+self.ndim], state[:, 1+self.ndim:]
			z_t = z_t[:, np.newaxis]
			K_t = P_t@self.H@np.linalg.inv(self.R)

			d_eta = np.zeros((self.ndim, self.ndim)) 
			if t > self.tau: # TODO warmup case?
				H_inv = np.linalg.inv(self.H)
				P_inv = np.linalg.solve(P_t.T@P_t + self.eps*np.eye(self.ndim), P_t.T)
				self.p_inv_t = P_inv

				tau_n = int(self.tau / self.dt)
				err_t, err_tau = err_hist[-1][:,np.newaxis], err_hist[-tau_n][:,np.newaxis]
				d_zz = (err_t@err_t.T - err_tau@err_tau.T) / self.tau
				self.e_zz_t = d_zz

				# if np.linalg.norm(d_zz) >= 1.0:
				# 	d_zz = np.zeros((self.ndim, self.ndim))

				# E_z = sum(err_hist[-tau_n:])[:,np.newaxis] / tau_n
				# d_uu = ((err_t - err_tau)/self.tau)@(E_z.T) + E_z@(((err_t - err_tau)/self.tau).T)
				# self.e_zz_t = d_zz - d_uu

				eta_new = H_inv@d_zz@H_inv.T@P_inv / 2
				if np.linalg.norm(eta_new) <= eta_bnd:
					d_eta = (eta_new - eta_t) / self.dt

			F_est = F_t - eta_t
			d_x = F_est@x_t + K_t@(z_t - self.H@x_t)
			d_P = F_est@P_t + P_t@F_est.T + self.Q - K_t@self.R@K_t.T
			d_state = np.concatenate((d_x, d_P, d_eta), axis=1)
			return d_state.ravel() # Flatten for integrator

		def g(t, state):
			return np.zeros((self.ode_ndim, self.ode_ndim))

		# state
		x0 = x0[:, np.newaxis]
		self.x_t = x0
		# covariance
		P0 = np.eye(self.ndim)
		self.P_t = P0
		# model variation
		eta0 = np.zeros((self.ndim, self.ndim))
		self.eta_t = eta0

		iv = np.concatenate((x0, P0, eta0), axis=1).ravel() # Flatten for integrator
		self.r = Integrator(f, g, self.ode_ndim)
		self.r.set_initial_value(iv, 0.)

	def load_vars(self, state: np.ndarray):
		state = state.reshape(self.ode_shape)
		x_t, P_t, eta_t = state[:, :1], state[:, 1:1+self.ndim], state[:, 1+self.ndim:]
		return x_t, P_t, eta_t

	def __call__(self, z_t: np.ndarray):
		''' Observe through filter ''' 
		self.r.set_f_params(z_t, self.err_hist, self.F(self.t))
		self.r.integrate(self.t + self.dt)
		x_t, P_t, eta_t = self.load_vars(self.r.y)
		self.x_t, self.P_t, self.eta_t = x_t, P_t, eta_t
		x_t = np.squeeze(x_t)
		err_t = z_t - x_t@self.H.T
		self.err_hist.append(err_t)
		return x_t.copy(), err_t # x_t variable gets reused somewhere...

	@property
	def ode_shape(self):
		return (self.ndim, 1 + 2*self.ndim) # representational dimension: x_t, P_t, eta_t

	@property
	def ode_ndim(self):
		return self.ode_shape[0] * self.ode_shape[1] # for raveled representation

	def f_eta(self, eta: np.ndarray):
		""" density function of variations """
		return stats.multivariate_normal.pdf(eta.ravel(), mean=self.eta_mu*np.ones(4), cov=self.eta_var*np.ones(4))

if __name__ == '__main__':
	import matplotlib.pyplot as plt

	set_seed(9001)

	dt = 1e-4
	T = 100.

	""" Noisy LTI example """ 
	z = Oscillator(dt, 0.0, 1.0)
	eta_mu, eta_var = 0., 0.03
	eta0 = np.random.normal(eta_mu, eta_var, (2, 2))
	eta = lambda t: eta0
	F_hat = lambda t: z.F(t) + eta(t)

	print(F_hat(0))
	f1 = KF(z.x0, F_hat, z.H, z.Q, z.R, dt)
	f2 = LKF(z.x0, F_hat, z.H, z.Q, z.R, dt, tau=0.25, eps=1e-2)

	max_err = 2.
	max_eta_err = 100
	max_zz = 100. 

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

		# Error condition 2
		if np.linalg.norm(f2.eta_t - eta(z.t)) > max_eta_err:
			print('Variation error overflowed!')
			break

		# Error condition 3
		if np.linalg.norm(f2.e_zz_t) > max_zz:
			print('d_zz overflowed!')
			break

	# start, end = None, 20000 # for case analysis
	start, end = None, None # for case analysis
	every = 10

	hist_t = np.array(hist_t)[start:end:every]
	hist_z = np.array(hist_z)[start:end:every]
	hist_f1_x = np.array(hist_f1_x)[start:end:every]
	hist_f2_x = np.array(hist_f2_x)[start:end:every]
	hist_f1_err = np.array(hist_f1_err)[start:end:every]
	hist_f2_err = np.array(hist_f2_err)[start:end:every]
	hist_eta = np.array(hist_eta)[start:end:every]

	# pdb.set_trace()

	fig, axs = plt.subplots(2, 5, figsize=(20, 20))
	
	axs[0,0].plot(hist_z[:,0], hist_z[:,1], color='blue', label='obs')
	axs[0,0].plot(hist_f1_x[:,0], hist_f1_x[:,1], color='orange', label='est')
	axs[0,0].legend()
	axs[0,0].set_title('System')

	axs[1,0].plot(hist_z[:,0], hist_z[:,1], color='blue', label='obs')
	axs[1,0].plot(hist_f2_x[:,0], hist_f2_x[:,1], color='orange', label='est')
	axs[1,0].legend()

	axs[0,1].plot(hist_t, hist_z[:,0], color='blue', label='obs')
	axs[0,1].plot(hist_t, hist_f1_x[:,0], color='orange', label='est')
	axs[0,1].set_title('Axis 1')

	axs[1,1].plot(hist_t, hist_z[:,0], color='blue', label='obs')
	axs[1,1].plot(hist_t, hist_f2_x[:,0], color='orange', label='est')

	axs[0,2].plot(hist_t, hist_z[:,1], color='blue', label='obs')
	axs[0,2].plot(hist_t, hist_f1_x[:,1], color='orange', label='est')
	axs[0,2].set_title('Axis 2')

	axs[1,2].plot(hist_t, hist_z[:,1], color='blue', label='obs')
	axs[1,2].plot(hist_t, hist_f2_x[:,1], color='orange', label='est')

	axs[0,3].plot(hist_t, hist_f1_err[:,0])
	axs[0,3].set_title('Axis 1 error')

	axs[1,3].plot(hist_t, hist_f2_err[:,0])

	axs[0,4].plot(hist_t, hist_f1_err[:,1])
	axs[0,4].set_title('Axis 2 error')

	axs[1,4].plot(hist_t, hist_f2_err[:,1])

	# var_err_rast = var_err.reshape((var_err.shape[0], 4))
	# axs[1,3].plot(hist_t, var_err_rast[:,0])
	# axs[1,3].plot(hist_t, var_err_rast[:,1])
	# axs[1,3].plot(hist_t, var_err_rast[:,2])
	# axs[1,3].plot(hist_t, var_err_rast[:,3])
	# axs[1,3].set_title('Variation error (rasterized)')

	plt.setp(axs[0, 0], ylabel='KF')
	plt.setp(axs[1, 0], ylabel='LKF')

	plt.show()
