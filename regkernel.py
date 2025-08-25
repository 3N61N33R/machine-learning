""" regkernel.py """
# %% 
import matplotlib.pyplot as plt
import numpy as np 

# x = np.array([[0.05, 0.2, 0.5, 0.75, 1.]]).T
# y = np.array([[0.4, 0.2, 0.6, 0.7, 1.]]).T

# n = x.shape[0]
# p = 1
# ngamma = (1-p)/p

# m = 1

# def k(x1, x2):
#     return np.ndarray.item(x1*x2 + x1*x1*x2*x2 + (x1**3)*(x2**3))

# def q1(x):
#     return 1
# q = [q1]

def kernel_train(k, m, q, ngamma, n, x, y):
    K = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            K[i,j] = k(x[i], x[j])
    Q = np.zeros((n,m))
    for i in range(n):
        for j in range(m):
            Q[i,j] = q[j](x[i])
    M1 = np.hstack((K @ K.T + (ngamma * K), K @ Q)) 
    M2 = np.hstack((Q.T @ K.T, Q.T @ Q))
    M = np.vstack((M1,M2))
    c = np.vstack((K, Q.T)) @ y
    ad = np.linalg.solve(M,c)
    return ad

### plotting the line
# xx = np.arange(0,1+0.01,0.01).reshape(-1,1)
# N = np.shape(xx)[0]

# g = np.zeros_like(xx)
# Qx = np.zeros((N,m))
# for i in range(N):
#     for j in range(m):
#         Qx[i,j] = q[j](xx[i])

# Kx = np.zeros((n,N))
# for i in range(n):
#     for j in range(N):
#         Kx[i,j] = k(x[i], xx[j])
# ad = kernel_train(k, m, q, ngamma, n, x, y)

# g = g + np.hstack((Kx.T, Qx)) @ ad

# plt.ylim((0,1.15))
# plt.plot(xx, g, label = 'p = {}'.format(p), linewidth = 2)
# plt.plot(x,y, 'b.', markersize=15)
# plt.xlabel('$x$')
# plt.ylabel('$y$')
# plt.legend()
