import numpy as np
from scipy.stats import multivariate_normal


def exp_max(Iter, K, pdf, train, Xmat, W_Init, P_Init):

    n, D = Xmat.shape
    p = np.zeros((K, n))
    W, P = W_Init, P_Init
    for i in range(0, Iter):
        # E-Step
        for k in range(0, K):
            p[k, :] = W[0, k] * pdf(P[:, k], Xmat)

        # M-Step
        p = p / sum(p, 0)
        W = np.mean(p, 1).reshape(1, 3)
        for k in range(0, K):
            P[:, k] = train(p[k, :], Xmat)
    return W, P, p


def normal_train(p, Xmat):
    m = (Xmat.T @ p.T) / sum(p)
    return m


def normal_pdf(m, Xmat):
    var = 1
    C = np.zeros((2, 2))
    C[0, 0] = var
    C[1, 1] = var
    mvn = multivariate_normal(m.T, C)
    return mvn.pdf(Xmat)


# Xmat = np.genfromtxt('clusterdata.csv', delimiter=',')
# W = np.array([[1/3,1/3,1/3]])
# M  = np.array([[-2.0,-4,0],[-3,1,-1]])
# W_final, P, p = exp_max(100, 3, normal_pdf, normal_train, Xmat, W, M)
