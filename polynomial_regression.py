"""polynomial_regression.py"""

import numpy as np
from numpy.linalg import norm, solve
import matplotlib.pyplot as plt
from numpy.random import rand, randn


def generate_data(p, beta, sig, n):
    """
     Generates data for polynomial regression.

     Args:
       p (int): Degree of the polynomial for data generation.
       beta (np.array): A column vector of the true polynomial coefficients [b_0, b_1, ..., b_p].T.
       sig (float): The standard deviation of the Gaussian noise (spread of the data).
       n (int): The number of data points to generate.

    Returns:
      tuple: A tuple containing:
        - u (np.array): A column vector of the independent variable values.
        - y (np.array): A column vector of the dependent variable values.

    """
    # Generate n feature values uniformly from [0, 1]
    u = np.random.rand(n, 1)

    # Generate the dependent variable values using the polynomial model with added Gaussian noise
    y = (u ** np.arange(0, p + 1)) @ beta + sig * np.random.randn(n, 1)
    return u, y


beta = np.array([[10, -140, 400, -250]]).T
n = 100
sig = 5
u, y = generate_data(3, beta, sig, n)


def model_matrix(p, u):
    """
    Constructs the design matrix (model matrix) for polynomial regression.

    Args:
       p (int): The degree of the polynomial model to be fitted.
       u (np.array): An (n, 1) array of feature values.

    Returns:
       np.array: An (n, p+1) design matrix X, where each column is u^j for j=0, ..., p.
    """
    n = len(u)  # fixed bug, n was undefined

    X = np.ones((n, 1))
    p_range = np.arange(0, p + 1)
    for p_current in p_range:
        if p_current > 0:
            X = np.hstack((X, u ** (p_current)))
    return X


def train(X, y):
    """
    Trains a linear regression model using the normal equation.

    Args:
        X (np.array): The design matrix of shape (n, p+1).
        y (np.array): The target values of shape (n, 1).

    Returns:
        np.array: The estimated coefficients (betahat) of shape (p+1, 1).
    """
    betahat = solve(X.T @ X, X.T @ y)
    return betahat


X, betahat = {}, {}
ps = [1, 3, 15]
for p in ps:
    X[p] = model_matrix(p, u)
    betahat[p] = train(X[p], y)


def test_coefficients(n, betahat, X, y):
    """
    Calculates the Mean Squared Error (MSE) loss for a given model.

    Args:
        n (int): The number of data points.
        betahat (np.array): The estimated model coefficients.
        X (np.array): The design matrix.
        y (np.array): The true target values.

    Returns:
        float: The mean squared error loss.
    """

    y_hat = X @ betahat
    loss = norm(y - y_hat) ** 2 / n
    return loss


# Varying the order of the polynomial and testing the model
# Generate new data for testing

beta_true_A = np.array([[5, 10]]).T
u_train, y_train = generate_data(1, beta_true_A, sig, n)

u_test, y_test = generate_data(1, beta, sig, n)


X_test = {}
training_loss = {}
test_loss = {}
for p in ps:
    X_test[p] = model_matrix(p, u_test)
    training_loss[p] = test_coefficients(n, betahat[p], X[p], y)
    test_loss[p] = test_coefficients(n, betahat[p], X_test[p], y_test)


# print the losses to see them
print("Training Loss:", training_loss)
print("Test Loss:", test_loss)

# Plot the points and true line and store in the list "plots"
xx = np.arange(np.min(u), np.max(u) + 5e-3, 5e-3)
yy = np.polyval(np.flip(beta), xx)
plots = [plt.plot(u, y, "k.", markersize=8)[0], plt.plot(xx, yy, "k--", linewidth=3)[0]]
# add the three curves
for i in ps:
    yy = np.polyval(np.flip(betahat[i]), xx)
    plots.append(plt.plot(xx, yy)[0])
plt.xlabel(r"$u$")
plt.ylabel(r"$y$")
plt.show()
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
