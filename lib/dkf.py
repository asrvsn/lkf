''' Discrete-time Kalman filter
'''

from systems import *
from utils import set_seed, diff_to_transferop
from lib.integrator import Integrator

from typing import Callable
import numpy as np

class DKF:
	def __init__(self, x0: np.ndarray, F: Callable, H: np.ndarray, Q: np.ndarray, R: np.ndarray, dt: float):
		self.F = F
		self.H = H
		self.Q = Q
		self.R = R
		self.t = 0
		self.dt = dt
		self.ndim = x0.shape[0]
		rep_ndim = self.ndim*(self.ndim+1) # representational dimension

		def f(t, x_t, P_t, z_t):
			F_t = self.F(t)
			M_t = self.H@P_t@self.H.T + self.R
			K_t = P_t@self.H.T@np.linalg.inv(M_t)
			x_t = F_t@x_t + F_t@K_t@(z_t - self.H@x_t)
			P_t = (F_t@(F_t@(P_t - K_t@self.H@P_t)).T).T + self.Q
			return x_t, P_t
		self.f = f

		x0 = x0[:, np.newaxis]
		P0 = np.eye(self.ndim)
		self.x_t = x0
		self.P_t = P0

	def __call__(self, z_t: np.ndarray):
		''' Observe through filter ''' 
		self.t += self.dt
		self.x_t, self.P_t = self.f(self.t, self.x_t, self.P_t, z_t[:, np.newaxis])
		x_t = np.squeeze(self.x_t)
		err_t = z_t - x_t@self.H.T
		return x_t.copy(), err_t # x_t variable gets reused somewhere...

if __name__ == '__main__':
	import matplotlib.pyplot as plt

	set_seed(6008)

	dt = 1e-3
	n = 20000
	z = Oscillator(dt, 0.0, 1.0) 
	# z = SpiralSink(dt, 0.0, 1.0)
	eta_mu, eta_var = 0., 0.0
	eta = np.random.normal(eta_mu, eta_var, (2, 2))
	F0 = diff_to_transferop(z.F(0) * dt)
	F_hat = lambda t: F0 + eta

	# z = TimeVarying(dt, 0.0, 1.0, f=1/20)
	# F_hat = lambda t: z.F(0)
	# eta = lambda t: F_hat(t) - z.F(t)

	print(F_hat(0))
	f = DKF(z.x0, F_hat, z.H, z.Q, z.R, dt)

	hist_t = []
	hist_z = []
	hist_x = []
	hist_err = []
	hist_p = []

	for _ in range(n):
		z_t = z()
		x_t, err_t = f(z_t)
		hist_z.append(z_t)
		hist_t.append(z.t)
		hist_x.append(x_t) 
		hist_err.append(err_t)
		hist_p.append(f.P_t.copy())

	start, end = None, 20000 # for case analysis
	# start, end = None, None # for case analysis

	hist_t = np.array(hist_t)[start:end]
	hist_z = np.array(hist_z)[start:end]
	hist_x = np.array(hist_x)[start:end]
	hist_err = np.array(hist_err)[start:end]
	hist_p = np.array(hist_p)[start:end]

	fig, axs = plt.subplots(2, 5, figsize=(20, 10))
	fig.suptitle('KF')

	axs[0,0].plot(hist_z[:,0], hist_z[:,1], color='blue', label='obs')
	axs[0,0].plot(hist_x[:,0], hist_x[:,1], color='orange', label='est')
	axs[0,0].legend()
	axs[0,0].set_title('System')
	axs[0,1].plot(hist_t, hist_z[:,0], color='blue', label='obs')
	axs[0,1].plot(hist_t, hist_x[:,0], color='orange', label='est')
	axs[0,2].plot(hist_t, hist_z[:,1], color='blue', label='obs')
	axs[0,2].plot(hist_t, hist_x[:,1], color='orange', label='est')
	axs[0,3].plot(hist_t, hist_err[:,0])
	axs[0,3].set_title('Axis 1 error')
	axs[0,4].plot(hist_t, hist_err[:,1])
	axs[0,4].set_title('Axis 2 error')

	p_rast = hist_p.reshape((hist_p.shape[0], 4))
	axs[1,0].plot(hist_t, p_rast[:,0])
	axs[1,0].plot(hist_t, p_rast[:,1])
	axs[1,0].plot(hist_t, p_rast[:,2])
	axs[1,0].plot(hist_t, p_rast[:,3])
	axs[1,0].set_title('P (rasterized)')

	hist_pin = np.zeros_like(hist_p)
	for i in range(hist_p.shape[0]):
		hist_pin[i] = np.linalg.inv(hist_p[i])
	p_inv_rast = hist_pin.reshape((hist_pin.shape[0], 4))
	axs[1,1].plot(hist_t, p_inv_rast[:,0])
	axs[1,1].plot(hist_t, p_inv_rast[:,1])
	axs[1,1].plot(hist_t, p_inv_rast[:,2])
	axs[1,1].plot(hist_t, p_inv_rast[:,3])
	axs[1,1].set_title('P_inv (rasterized)')

	plt.show()

