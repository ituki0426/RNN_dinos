import numpy as np
def softmax(x):
    """
    Computes the softmax activation function for a given input array.

    Parameters:
        x (ndarray): Input array.

    Returns:
        ndarray: Array of the same shape as `x`, containing the softmax activation values.
    """
    # shift the input to prevent overflow when computing the exponentials
    x = x - np.max(x)
    # compute the exponentials of the shifted input
    p = np.exp(x)
    # normalize the exponentials by dividing by their sum
    return p / np.sum(p)
