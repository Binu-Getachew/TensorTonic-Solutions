



import numpy as np

def percentiles(x, q):
    """
    Compute percentiles using linear interpolation.
    """
    sorted_x = np.sort(x)  # Avoid using the built-in keyword 'input'
    return np.percentile(sorted_x, q)
