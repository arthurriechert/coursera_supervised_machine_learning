import numpy as np

def sigmoid (z):
    return 1.0 / (1.0 + np.exp(-z))

def compute_model (W, X, b):
    
    z = np.dot(X, W) + b
    a = sigmoid(z)

    return a

def compute_cost(W, X, b, Y):
    m = X.shape[0]
    prediction = compute_model(W, X, b)
    epsilon = 1e-7
    cost = (-1/m) * np.sum(Y*np.log(prediction + epsilon) + (1-Y)*np.log(1-prediction + epsilon))
    return cost

X_train = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])  #(m,n)
Y_train = np.array([0, 0, 0, 1, 1, 1])    

W_init = np.array([1,1])
b_init = -3

f_wb = compute_model(W_init, X_train, b_init)
J_wb = compute_cost(W_init, X_train, b_init, Y_train)

print(f"Results of Computation: {f_wb} | Cost of Computation: {J_wb}")