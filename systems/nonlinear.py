''' Nonlinear systems using Koopman operator ''' 

import torch
import numpy as np
from scipy.integrate import ode, solve_ivp
import matplotlib.pyplot as plt
from typing import Callable
from mpl_toolkits.mplot3d import Axes3D
import pdb

from utils import transferop_to_diff
from totorch.utils import set_seed
from totorch.features import PolynomialObservable
import totorch.operators as op
from totorch.predict import extrapolate

class HiddenProcess:
	def __init__(self, x0: np.ndarray, sys: Callable, F: Callable, proj: Callable, dt: float, H: np.ndarray, var_v: float, ndim: int):
		''' Arbitrary diff.eq with observation noise ''' 
		self.x0 = x0
		self.r = ode(sys).set_integrator('dopri5').set_initial_value(x0)
		self.dt = dt
		self.F = F
		self.proj = proj
		self.H = H
		self.var_v = var_v
		# For compatibility
		self.ndim = ndim
		self.R = np.eye(ndim) * var_v
		self.Q = np.eye(ndim) * 0.

	def __call__(self):
		''' Observe process '''
		self.r.integrate(self.t + self.dt)
		x_t = np.array(self.r.y)
		y_t = self.H@self.proj(x_t)
		v_t = self.var_v * np.random.normal(0.0, np.sqrt(self.dt), y_t.shape[0])
		z_t = y_t + v_t
		return z_t 

	@property
	def t(self):
		return self.r.t

class VanDerPol(HiddenProcess):
	def __init__(self, dt: float, var_v: float, mu: Callable=None):
		if mu == None:
			mu = lambda t: 3.0
		sys = lambda t, z: [z[1], mu(t)*(1-z[0]**2)*z[1] - z[0]]
		x0 = np.array([1, 0])

		# Init features
		p, d, k = 4, 2, 8
		self.obs = PolynomialObservable(p, d, k)
		proj = lambda x: self.obs.call_numpy(x)
		H = np.eye(k)

		# Fit linear model
		a, b = 0, 20
		self.t_fit = np.linspace(a, b, int(b/dt))
		x_fit = solve_ivp(sys, [a, b], x0, t_eval=self.t_fit).y
		self.x_fit = torch.from_numpy(x_fit).float()
		self.K = op.solve(self.x_fit, obs=self.obs)
		assert not torch.isnan(self.K).any().item(), 'Got NaN in the model!'

		# Use K as discrete-time model
		koop = op.Koopman(self.K.numpy(), self.obs)
		F = lambda t: koop

		super().__init__(x0, sys, F, proj, dt, H, var_v, k)

	def show_model(self):
		koop = op.Koopman(self.K.numpy(), self.obs)
		test_x = self.obs(self.x_fit)[:, :3].numpy()
		print(koop@test_x)
		pdb.set_trace()
		print(test_x[np.newaxis]@koop.T)

		pred_x = extrapolate(self.x_fit[:,0], self.K, self.obs, self.x_fit.shape[1]-1, unlift_every=False)
		err = torch.norm(self.x_fit - pred_x, dim=0).cpu().numpy()
		X = self.x_fit.cpu().numpy()
		pred_X = pred_x.cpu().numpy()

		fig, axs = plt.subplots(1, 3)
		axs[0].plot(X[0], X[1])
		axs[0].set_title('baseline')
		axs[1].plot(pred_X[0], pred_X[1])
		axs[1].set_title('predicted')
		axs[2].plot(np.arange(len(err)), err)
		axs[2].set_title('error')

		plt.tight_layout()
		plt.show()

class Lorenz(HiddenProcess):
	def __init__(self, dt: float, var_v: float, sigma: Callable=None, beta: Callable=None, rho: Callable=None):
		if sigma is None: sigma = lambda t: 10
		if beta is None: beta = lambda t: 2.667
		if rho is None: rho = lambda t: 28

		def sys(t, z):
			[u, v, w] = z
			du = -sigma(t)*(u-v)
			dv = rho(t)*u - v - u*w
			dw = -beta(t)*w + u*v
			return np.array([du, dv, dw])

		x0 = np.array([0, 1, 1.05])

		# Init features
		p, d, k = 2, 3, 9
		self.obs = PolynomialObservable(p, d, k)
		proj = lambda x: self.obs.call_numpy(x)
		H = np.eye(k)

		# Fit linear model
		a, b = 0, 30
		self.t_fit = np.linspace(a, b, int(b/dt))
		x_fit = solve_ivp(sys, [a, b], x0, t_eval=self.t_fit).y
		self.x_fit = torch.from_numpy(x_fit).float()
		self.K = op.solve(self.x_fit, obs=self.obs)
		assert not torch.isnan(self.K).any().item(), 'Got NaN in the model!'

		# Use K as discrete-time model
		koop = op.Koopman(self.K.numpy(), self.obs)
		F = lambda t: koop

		super().__init__(x0, sys, F, proj, dt, H, var_v, k)

	def show_model(self):
		pred_x = extrapolate(self.x_fit[:,0], self.K, self.obs, self.x_fit.shape[1]-1)
		err = torch.norm(self.x_fit - pred_x, dim=0).cpu().numpy()
		X = self.x_fit.cpu().numpy()
		pred_X = pred_x.cpu().numpy()

		fig = plt.figure()
		ax = fig.add_subplot(1, 3, 1, projection='3d')
		ax.plot(X[0], X[1], X[2])
		ax.set_title('baseline')

		ax = fig.add_subplot(1, 3, 2, projection='3d')
		ax.plot(pred_X[0], pred_X[1], pred_X[2])
		ax.set_title('predicted')

		ax = fig.add_subplot(1, 3, 3)
		ax.plot(np.arange(len(err)), err)
		ax.set_title('error')

		plt.tight_layout()
		plt.show()


if __name__ == '__main__':
	set_seed(9001)

	z = VanDerPol(1e-4, 1.0)
	z.show_model()

	# z = Lorenz(5e-3, 1.0)
	# z.show_model()