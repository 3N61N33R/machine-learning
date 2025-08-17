""" hereditary_traits.py """
import numpy as np

def generate_data(mean, var, hrd, n_each, n):
   g = mean + np.sqrt(hrd*var)*np.random.randn(n,1)
   gs = np.tile(g, (1, n_each))
   x = gs + np.sqrt((1-hrd)*var)*np.random.randn(n,n_each)
   return g, x

#g, x = generate_data(175, 100, 0.8, 2, 1000)
