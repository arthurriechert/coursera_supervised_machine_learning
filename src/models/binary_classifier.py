import numpy as np

def sigmoid (z):
    return 1.0 / (1.0 + np.exp(-z))

def compute_model (W, X, b):
    
    z = np.dot(W.T, X) + b
    a = sigmoid(z)

    return z

X_train = np.array([[1.5, 5.3, 3],
                   [3, 2, 5],
                   [1, 2, 9]], dtype=np.float32)

W_init = np.array([1, -2, -100], dtype=np.float32)
b_init = 3

f_wb = compute_model(W_init, X_train, b_init)

print(f"Results of Computation: {compute_model(W_init, X_train, b_init)}.")