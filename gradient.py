# Kesar TN
# University of Central Florida
# kesar@Knights.ucf.edu
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Function to read input
def get_data(input):
    # Read the data.txt file using pandas
    data = pd.read_fwf(input)
    data.columns = ["non", "X1", "X2", "X3", "X4", "X5"]
    # Only create separate numpy arrays for X and b
    X = data[['X1', 'X2', 'X3', 'X4', 'X5']][1:21].to_numpy()
    b = data['X1'][25:].to_numpy()

    return X, b.reshape((20,1))

# Function for the iterative gradient decent algorithm
def gradient_descent(h, X, b, learning_rate, iterations):
    # Initiate values
    h_deriv = 0
    h_plot = []
    N = len(X)
    # Loop untile iterations
    for _ in range(iterations):
        # 2*X'(Xh - b) ----- Calculate partial derivatives
        h_deriv = 2 * X.T @ (X @ h - b)

        # Update h value though the first order partial derivative
        h -= (h_deriv / float(N)) * learning_rate
        iterative_e = (X @ h) - b
        # Plot the error values by using a list
        temp = iterative_e.T @ iterative_e
        h_plot.append(temp[0][0])
    
    h_plot = np.asarray(h_plot)
    plt.plot(h_plot)
    plt.show()

    return h

# Call main Function
if __name__ == "__main__":
    
    # Create data values and randomize h
    X, b = get_data('data.txt')
    h = np.random.rand(5, 1)

    # Calculate mean squared h using ((X * X')^-1 * X') * b
    mse_h = (np.linalg.inv(X.T @ X) @ X.T) @ b
    # print(mse_h)

    # Call gradient_descent Function
    iterative_h = gradient_descent(h, X, b, 0.001, 1000)
    # print(updated_h)

    # Find both e values by the loss function provided
    iterative_e = (X @ iterative_h) - b
    mse_e = (X @ mse_h) - b
    print('Iterative gradient decent e: ', iterative_e)
    print('Mean Squared error r', mse_e)

    # We plot both the errors to compare
    plt.plot(mse_e)
    plt.plot(iterative_e)
    plt.show()