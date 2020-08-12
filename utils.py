import numpy as np
import random
import torch
import scipy.linalg as linalg

def set_seed(seed=None):
	random.seed(seed)
	np.random.seed(seed)
	if seed is None: 
		torch.manual_seed(random.randint(1,1e6))
	else:
		torch.manual_seed(seed)

def rms(arr): 
	return np.sqrt(np.mean(arr ** 2))

def diff_to_transferop(A: np.ndarray):
	return linalg.expm(A)

def transferop_to_diff(A: np.ndarray):
	return np.real(linalg.logm(A, disp=False)[0])