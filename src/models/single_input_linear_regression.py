import numpy as np
import random as rand
import matplotlib.pyplot as plt

"""
Variations from Andrew Ng's code.


Course code utilized for loops to perform calculations

I made operations using matrices to take advantage of numpy's speed.
"""

# Generate a close approximation for W, b
def wb_init():
    return rand.randint(100, 200), rand.randint(0, 200)

# Create prediction with simple linear regression model
def compute_model_output(X, w, b):
    return w * X + b

# Calculate cost using the Mean Squared Error
def compute_cost(X, Y, w, b):
    m = X.shape[0]
    prediction = compute_model_output(X, w, b)
    cost = np.sum((prediction - Y) ** 2) / (2 * m)
    return cost, prediction

# Find the gradient using respective partial derivatives
def compute_gradient(X, Y, w, b):
    f_wb = compute_model_output(X, w, b)
    m = X.shape[0]
    dj_dw = (np.sum((f_wb - Y) * X)) / m
    dj_db = (np.sum((f_wb - Y))) / m

    return dj_dw, dj_db

# Update w and b according to alpha
def gradient_descent(X, Y, w, b, alpha):

    dj_dw, dj_db = compute_gradient(X, Y, w, b)

    w = w - alpha * dj_dw
    b = b - alpha * dj_db

    return w, b

# Contour and scatter plot
def test():
    # Generate synthetic training data
    x_train = np.linspace(0, 10, 100)
    y_train = 150 * x_train + 75 + np.random.normal(0, 20, x_train.shape)  # add some noise

    # Initialize w, b
    w, b = wb_init()

    # Learning rate
    alpha = 0.01

    # Number of iterations
    n_iterations = 1000

    # Store the cost and parameter values
    cost_values = []
    w_values = []
    b_values = []

    # Perform gradient descent
    for i in range(n_iterations):
        w, b = gradient_descent(x_train, y_train, w, b, alpha)
        cost, _ = compute_cost(x_train, y_train, w, b)
        cost_values.append(cost)
        w_values.append(w)
        b_values.append(b)

    # Prepare grid for contour plot
    w_grid = np.linspace(min(w_values)-10, max(w_values)+10, 100)
    b_grid = np.linspace(min(b_values)-10, max(b_values)+10, 100)
    W, B = np.meshgrid(w_grid, b_grid)
    Z = np.zeros(W.shape)

    # Calculate cost function for each point in the grid
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            Z[i, j], _ = compute_cost(x_train, y_train, W[i, j], B[i, j])

    # Create contour plot
    plt.figure(figsize=(10, 6))
    contour = plt.contour(W, B, Z, levels=np.logspace(-2, 3, 20), cmap='viridis')
    plt.clabel(contour, inline=1, fontsize=10)
    plt.plot(w_values, b_values, 'r.')  # Plot the path of gradient descent
    plt.title('Contour plot of cost function')
    plt.xlabel('w')
    plt.ylabel('b')
    plt.grid(True)
    plt.show()

    # Create scatter plot of data and regression line
    plt.figure(figsize=(10, 6))
    plt.scatter(x_train, y_train, color='blue', label='Data points')
    plt.plot(x_train, w * x_train + b, color='red', label=f'y = {w:.2f}x + {b:.2f}')
    plt.title('Data and regression line')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.show()
