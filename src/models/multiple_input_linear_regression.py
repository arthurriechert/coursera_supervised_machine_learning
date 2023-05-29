import numpy as np
import random as rand
import math, copy


"""
Code from Week 2 of Supervised Learning by Andrew Ng on Coursera
"""

# Generate random values for w and b
# Generate random values for w and b
def init_wb(j):
    return np.random.rand(4), rand.randint(400, 2000)

# Generate a prediction
def predict(x, w, b): 
    p = np.dot(x, w) + b     
    return p 

# Calculate the cost of predictions use MSE
def compute_cost(X, y, w, b): 
    m = X.shape[0]
    cost = 0.0
    for i in range(m):                                
        f_wb_i = np.dot(X[i], w) + b           #(n,)(n,) = scalar (see np.dot)
        cost = cost + (f_wb_i - y[i])**2       #scalar
    cost = cost / (2 * m)                      #scalar    
    return cost

# Calculate gradient using partial derivatives
def compute_gradient(X, y, w, b): 
    m,n = X.shape           #(number of examples, number of features)
    dj_dw = np.zeros((n,))
    dj_db = 0.

    for i in range(m):                             
        err = (np.dot(X[i], w) + b) - y[i]   
        for j in range(n):                         
            dj_dw[j] = dj_dw[j] + err * X[i, j]    
        dj_db = dj_db + err                        
    dj_dw = dj_dw / m                                
    dj_db = dj_db / m                                
        
    return dj_db, dj_dw

# Update w and b according to alpha
def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters): 
    # An array to store cost J and w's at each iteration primarily for graphing later
    J_history = []
    w = copy.deepcopy(w_in)  #avoid modifying global w within function
    b = b_in
    
    for i in range(num_iters):

        # Calculate the gradient and update the parameters
        dj_db,dj_dw = gradient_function(X, y, w, b)   ##None

        # Update Parameters using w, b, alpha and gradient
        w = w - alpha * dj_dw               ##None
        b = b - alpha * dj_db               ##None
      
        # Save cost J at each iteration
        if i<100000:      # prevent resource exhaustion 
            J_history.append( cost_function(X, y, w, b))

        # Print cost every at intervals 10 times or as many iterations if < 10
        if i% math.ceil(num_iters / 10) == 0:
            print(f"Iteration {i:4d}: Cost {J_history[-1]:8.2f}   ")
        
    return w, b, J_history #return final w,b and J history for graphing

# Test for our model
def test_1():
    # Training data
    X_train = np.array([[2104, 5, 1, 45],
                        [1416, 3, 2, 40],
                        [852, 2, 1, 35]])
    y_train = np.array([460, 232, 178])

    b_init = 785.1811367994083
    w_init = np.array([ 0.39133535, 18.75376741, -53.36032453, -26.42131618])
    print(f"w_init shape: {w_init.shape}, b_init type: {type(b_init)}")

    # initialize parameters
    initial_w = np.zeros_like(w_init)
    initial_b = 0.
    # some gradient descent settings
    iterations = 1000
    alpha = 5.0e-7
    # run gradient descent 
    w_final, b_final, J_hist = gradient_descent(X_train, y_train, initial_w, initial_b,
                                                        compute_cost, compute_gradient, 
                                                        alpha, iterations)
    print(f"b,w found by gradient descent: {b_final:0.2f},{w_final} ")
    m,_ = X_train.shape
    for i in range(m):
        print(f"prediction: {np.dot(X_train[i], w_final) + b_final:0.2f}, target value: {y_train[i]}")

# Running all tests
test_1()
"""
test_2()
test_3()
test_4()
"""
