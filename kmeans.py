import numpy as np

def kmeans(eps, K, Xmat, c_init):
    n, D = Xmat.shape
    c = c_init
    c_old = np.zeros(c.shape)
    dist2 = np.zeros((K,n))
    while np.abs(c - c_old).sum() > eps:
        c_old = c.copy()
        for i in range(0,K): #compute the squared distances
            dist2[i,:] = np.sum((Xmat - c[:,i].T)**2, 1)        
        label = np.argmin(dist2,0) #assign the points to nearest centroid
        minvals = np.amin(dist2,0)
        for i in range(0,K): # recompute the centroids
            entries = np.where(label == i)
            c[:,i] = np.mean(Xmat[entries,:], 1).reshape(1,2)
    return c, label

#Xmat = np.genfromtxt('clusterdata.csv', delimiter=',')
#c_init  = np.array([[-2.0,-4,0],[-3,1,-1]])
#eps = 0.001
#K = 3
#c, label = kmeans(eps, K, Xmat, c_init) 
