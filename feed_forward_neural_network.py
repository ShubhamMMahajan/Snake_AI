import numpy as np
from numba import jit

#number of nodes per layer
n_x = 6
n_h = 9
n_h2 = 15
n_y = 4

#used to determine number of weights from one layer to the next
W1_shape = (9,6)
W2_shape = (15,9)
W3_shape = (4,15)

#code is compiled using numba to increase computation speed
@jit(nopython=True, fastmath=True)
def get_weights_from_encoded(individual):
    W1 = individual[0:W1_shape[0] * W1_shape[1]]
    W2 = individual[W1_shape[0] * W1_shape[1]:W2_shape[0] * W2_shape[1] + W1_shape[0] * W1_shape[1]]
    W3 = individual[W2_shape[0] * W2_shape[1] + W1_shape[0] * W1_shape[1]:]
    return (
    W1.reshape(W1_shape[0], W1_shape[1]), W2.reshape(W2_shape[0], W2_shape[1]), W3.reshape(W3_shape[0], W3_shape[1]))

@jit(nopython=True, fastmath=True)
def softmax(z):
    s = np.exp(z.T) / np.sum(np.exp(z.T), axis=1).reshape(-1, 1)
    return s

@jit(nopython=True, fastmath=True)
def sigmoid(z):
    s = 1 / (1 + np.exp(-z))
    return s
    
def forward_propagation(X, individual):
    W1, W2, W3 = get_weights_from_encoded(individual)
    Z1 = np.dot(W1, X.T)
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2, A1)
    A2 = np.tanh(Z2)
    Z3 = np.dot(W3, A2)
    A3 = softmax(Z3)
    return A3

