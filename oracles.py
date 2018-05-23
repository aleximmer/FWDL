import numpy as np 
from scipy.sparse.linalg import svds


def LMO_l1(grad, kappa):
    shape = grad.shape
    grad = grad.reshape(-1)
    s = np.zeros(grad.shape)
    coord = np.argmax(np.abs(grad))
    s[coord] = kappa * np.sign(grad[coord])
    return - s.reshape(*shape)


def LMO_nuclear(grad, kappa):
    u, s, vt = svds(grad, k=1, which='LM')
    return - kappa * np.outer(u, vt)
