import numpy as np
def maxpool_forward(X, pool_size, stride):
    """
    Compute the forward pass of 2D max pooling.
    """
    X = np.array(X)
    H, W = X.shape
    

    out_h = (H - pool_size) // stride + 1
    out_w = (W - pool_size) // stride + 1
    
    output = np.zeros((out_h, out_w))

    for i in range(out_h):
        for j in range(out_w):
            curr_y= i * stride
            curr_x= j * stride

            window=X[curr_y:curr_y + pool_size, curr_x:curr_x + pool_size]
            output[i,j]=np.max(window)
    return output.tolist()