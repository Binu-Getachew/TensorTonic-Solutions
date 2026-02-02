import numpy as np

def leaky_relu(x, alpha=0.01):
    # Convert input to a numpy array to handle lists or scalars
    x_arr = np.asanyarray(x)
    
    # Vectorized operation: return x if x > 0, else alpha * x
    return np.where(x_arr > 0, x_arr, alpha * x_arr)