import numpy as np 
from projections import euclidean_proj_l1ball


def LMO_l1(grad, kappa):
    shape = grad.shape
    grad = grad.reshape(-1)
    s = np.zeros(grad.shape)
    coord = np.argmax(np.abs(grad))
    s[coord] = kappa * np.sign(grad[coord])
    return - s.reshape(*shape)


def P_l1(grad, kappa):
    shape = grad.shape
    grad = grad.reshape(-1)
    proj = euclidean_proj_l1ball(grad, kappa)
    return proj.reshape(*shape)
