import numpy as np 
from scipy.sparse.linalg import svds


def LMO_l1(grad, kappa):
    s = np.zeros(grad.shape)
    coord = np.argmax(np.abs(grad))
    s[coord] = kappa * np.sign(grad[coord])
    return - s


def LMO_nuclear(grad, kappa):
    u, s, vt = svds(grad, k=1, which='LM')
    return - kappa * np.outer(u, vt)
